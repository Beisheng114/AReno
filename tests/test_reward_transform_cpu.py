from __future__ import annotations

import logging
import math
import unittest
from types import SimpleNamespace

from areno.api.data import PromptItem
from areno.api.models import RolloutResult, RolloutSequence
from areno.api.reward_transform import RewardTransformConfig, transform_rewards
from areno.api.rewards import compute_group_advantages
from areno.api.trainers.policy_only import PolicyOnlyTrainer


class RewardTransformUnitTest(unittest.TestCase):
    """Pure-function tests for reward clipping and per-batch standardization."""

    def test_disabled_mode_is_numerically_unchanged(self):
        """Disabled mode must return the input list verbatim (same order/values)."""

        rewards = [1.0, 2.0, 3.0, -4.0, 0.5]
        transformed, summary = transform_rewards(rewards, RewardTransformConfig())

        self.assertEqual(transformed, rewards)
        self.assertEqual(summary["mode"], "disabled")
        # Raw and transformed distributions are reported separately but equal.
        self.assertEqual(summary["raw"], summary["transformed"])

    def test_clip_clamps_to_range(self):
        """Clip mode clamps every reward into [clip_min, clip_max]."""

        transformed, summary = transform_rewards(
            [1.0, 2.0, 3.0, -5.0, 9.0],
            RewardTransformConfig(mode="clip", clip_min=0.0, clip_max=2.0),
        )

        self.assertEqual(transformed, [1.0, 2.0, 2.0, 0.0, 2.0])
        self.assertEqual(summary["raw"]["min"], -5.0)
        self.assertEqual(summary["transformed"]["min"], 0.0)

    def test_standardize_produces_zero_mean_unit_std(self):
        """Per-batch standardization yields mean=0, std=1."""

        transformed, _ = transform_rewards(
            [1.0, 2.0, 3.0], RewardTransformConfig(mode="standardize")
        )

        self.assertAlmostEqual(sum(transformed), 0.0, places=6)
        self.assertAlmostEqual(
            math.sqrt(sum((x - sum(transformed) / 3) ** 2 for x in transformed) / 3),
            1.0,
            places=5,
        )

    def test_standardize_constant_rewards_returns_zeros(self):
        """Constant rewards have zero std; output must stay finite (all zeros)."""

        transformed, summary = transform_rewards(
            [3.0, 3.0, 3.0], RewardTransformConfig(mode="standardize")
        )

        self.assertEqual(transformed, [0.0, 0.0, 0.0])
        self.assertTrue(all(math.isfinite(x) for x in transformed))
        self.assertEqual(summary["transformed"]["std"], 0.0)

    def test_empty_input_returns_empty_with_empty_summary(self):
        """An empty reward batch is valid; stats report count=0 and None moments."""

        transformed, summary = transform_rewards([], RewardTransformConfig(mode="standardize"))

        self.assertEqual(transformed, [])
        self.assertEqual(summary["raw"]["count"], 0)
        self.assertEqual(summary["transformed"]["count"], 0)
        self.assertIsNone(summary["raw"]["mean"])

    def test_empty_input_emits_no_numpy_warnings(self):
        """Empty standardize input must not trigger numpy RuntimeWarnings."""

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # All three modes should handle empty input cleanly.
            for mode_cfg in [
                RewardTransformConfig(),
                RewardTransformConfig(mode="clip", clip_min=0.0, clip_max=1.0),
                RewardTransformConfig(mode="standardize"),
            ]:
                transformed, _ = transform_rewards([], mode_cfg)
                self.assertEqual(transformed, [])

    def test_nan_raw_reward_raises_with_position(self):
        """Non-finite raw rewards raise a clear error naming the stage and index."""

        with self.assertRaisesRegex(ValueError, "non-finite") as cm:
            transform_rewards([1.0, float("nan"), 3.0], RewardTransformConfig(mode="standardize"))
        self.assertIn("index 1", str(cm.exception))

    def test_inf_raw_reward_raises_in_disabled_mode(self):
        """Disabled mode still rejects non-finite inputs rather than propagating NaN."""

        with self.assertRaisesRegex(ValueError, "non-finite"):
            transform_rewards([1.0, float("inf")], RewardTransformConfig())

    def test_extreme_inputs_produce_finite_output_when_clipped(self):
        """Huge-magnitude rewards become finite after clipping."""

        transformed, _ = transform_rewards(
            [1e308, -1e308, 0.0],
            RewardTransformConfig(mode="clip", clip_min=-1.0, clip_max=1.0),
        )

        self.assertEqual(transformed, [1.0, -1.0, 0.0])
        self.assertTrue(all(math.isfinite(x) for x in transformed))

    def test_summary_keeps_raw_and_transformed_separate(self):
        """The two distribution blocks must differ under an actual transform."""

        _, summary = transform_rewards(
            [0.0, 10.0], RewardTransformConfig(mode="clip", clip_min=0.0, clip_max=1.0)
        )

        self.assertNotEqual(summary["raw"]["max"], summary["transformed"]["max"])
        self.assertEqual(summary["raw"]["max"], 10.0)
        self.assertEqual(summary["transformed"]["max"], 1.0)


class RewardTransformConfigTest(unittest.TestCase):
    """Config validation surfaces bad inputs early with a clear message."""

    def test_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            RewardTransformConfig(mode="bogus")

    def test_clip_requires_both_bounds(self):
        with self.assertRaisesRegex(ValueError, "requires both clip_min and clip_max"):
            RewardTransformConfig(mode="clip", clip_min=0.0)

    def test_clip_rejects_inverted_bounds(self):
        with self.assertRaisesRegex(ValueError, "clip_min .* must be <= clip_max"):
            RewardTransformConfig(mode="clip", clip_min=5.0, clip_max=1.0)

    def test_clip_rejects_non_finite_bounds(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            RewardTransformConfig(mode="clip", clip_min=float("-inf"), clip_max=1.0)

    def test_rejects_non_positive_eps(self):
        with self.assertRaisesRegex(ValueError, "eps must be a positive finite"):
            RewardTransformConfig(mode="standardize", eps=0.0)

    def test_enabled_property_reflects_mode(self):
        self.assertFalse(RewardTransformConfig().enabled)
        self.assertTrue(RewardTransformConfig(mode="clip", clip_min=0.0, clip_max=1.0).enabled)
        self.assertTrue(RewardTransformConfig(mode="standardize").enabled)


class PolicyOnlyMaterializeIntegrationTest(unittest.TestCase):
    """The policy-only trainer derives advantages from transformed rewards.

    The trainer is instantiated via ``object.__new__`` with a fake tokenizer
    and a ``SimpleNamespace`` config so the full rollout/backend/tensor-parallel
    stack is bypassed.  This isolates the reward-transform and advantage
    orchestration logic on CPU.

    Minimal GPU validation that remains (not covered here):
        * Under tensor-parallel training each rank shapes only its own reward
          shard, so ``standardize`` statistics are rank-local.  The documented
          per-rank semantics (see ``docs/cli/training.rst``) can only be
          observed on a multi-GPU run.
        * End-to-end GSPO/GRPO training with a real model, rollout engine, and
          backend — to confirm that transformed rewards flow through the loss
          without shape/dtype mismatches — requires CUDA hardware.
    """

    # Raw per-sample rewards for the single three-sample prompt below; the
    # reward_fn dispatches on sample_index so all samples share one prompt.
    _RAW_REWARDS = [1.0, 5.0, 9.0]

    def _reward_fn(self, record):
        idx = record.metadata["sample_index"]
        return self._RAW_REWARDS[idx]

    def _build_trainer(self, *, reward_transform_mode="disabled", clip_min=None, clip_max=None):
        policy = object.__new__(PolicyOnlyTrainer)
        policy.logger = logging.getLogger("test.reward_transform.materialize")
        policy.reward_fn = self._reward_fn
        cfg = RewardTransformConfig(
            mode=reward_transform_mode, clip_min=clip_min, clip_max=clip_max
        )
        policy.config = SimpleNamespace(reward_transform_config=lambda: cfg)
        return policy

    def _build_batch_and_results(self):
        # One prompt with three rollout samples; rewards [1, 5, 9] by sample idx.
        items = [PromptItem(prompt="a", solutions=None, input_tokens=[1], record={})]
        results = [
            RolloutResult(
                sequences=[
                    RolloutSequence(resp_tokens=[10 + i], resp_logprobs=[-0.1 - 0.1 * i])
                    for i in range(3)
                ]
            )
        ]
        return items, results

    def _pkt(self):
        class _Tokenizer:
            eos_token_id = 0

            def decode(self, tokens):
                return "x"

        return _Tokenizer()

    def _run(self, policy):
        items, results = self._build_batch_and_results()
        return policy._materialize_train_batch(self._pkt(), _make_prompt_batch(items), results)

    def test_disabled_materialize_matches_baseline_advantages(self):
        """Disabled mode advantages must equal compute_group_advantages on raw rewards."""

        policy = self._build_trainer()
        train_batch, rewards_all, _ = self._run(policy)

        baseline = compute_group_advantages(self._RAW_REWARDS)
        self.assertEqual(rewards_all, self._RAW_REWARDS)
        self.assertEqual([seq.advantages[-1] for seq in train_batch], baseline)
        # Raw reward preserved; transformed_reward unset in disabled mode.
        self.assertEqual([seq.reward for seq in train_batch], self._RAW_REWARDS)
        self.assertTrue(all(seq.transformed_reward is None for seq in train_batch))

    def test_clip_materialize_derives_advantages_from_clipped_rewards(self):
        """Advantages come from the clipped distribution; raw rewards are still logged raw."""

        clip_cfg = RewardTransformConfig(mode="clip", clip_min=0.0, clip_max=5.0)
        policy = self._build_trainer(
            reward_transform_mode="clip", clip_min=0.0, clip_max=5.0
        )
        train_batch, rewards_all, _ = self._run(policy)

        clipped, _ = transform_rewards(self._RAW_REWARDS, clip_cfg)
        expected = compute_group_advantages(clipped)
        self.assertEqual([seq.advantages[-1] for seq in train_batch], expected)
        # Raw rewards still returned and stamped on seq.reward unchanged.
        self.assertEqual(rewards_all, self._RAW_REWARDS)
        self.assertEqual([seq.reward for seq in train_batch], self._RAW_REWARDS)
        # transformed_reward records the value actually fed into advantages.
        self.assertEqual([seq.transformed_reward for seq in train_batch], clipped)

    def test_standardize_materialize_logs_and_sets_transformed_reward(self):
        """Standardize mode logs a structured summary and populates transformed_reward."""

        std_cfg = RewardTransformConfig(mode="standardize")
        policy = self._build_trainer(reward_transform_mode="standardize")
        with self.assertLogs(policy.logger, level="INFO") as logged:
            train_batch, rewards_all, _ = self._run(policy)

        standardized, _ = transform_rewards(self._RAW_REWARDS, std_cfg)
        expected = compute_group_advantages(standardized)
        self.assertEqual([seq.advantages[-1] for seq in train_batch], expected)
        self.assertEqual([seq.transformed_reward for seq in train_batch], standardized)
        self.assertTrue(any("stage=reward_transform" in line for line in logged.output))
        self.assertTrue(any("mode=standardize" in line for line in logged.output))
        # The log line must carry both raw and transformed distribution blocks.
        transform_line = next(line for line in logged.output if "stage=reward_transform" in line)
        self.assertIn("raw[", transform_line)
        self.assertIn("transformed[", transform_line)

    def test_disabled_mode_emits_no_transform_log(self):
        """Disabled mode must not produce reward_transform log lines (historic behavior)."""

        policy = self._build_trainer()
        records: list[str] = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Handler(level=logging.INFO)
        logger = logging.getLogger("test.reward_transform.materialize")
        logger.addHandler(handler)
        try:
            self._run(policy)
        finally:
            logger.removeHandler(handler)
        self.assertFalse(any("stage=reward_transform" in line for line in records))


def _make_prompt_batch(items):
    from areno.api.data import PromptBatch

    return PromptBatch(items=items, scanned=len(items), skipped_long=0, total_skipped_long=0)


if __name__ == "__main__":
    unittest.main()