from __future__ import annotations

import unittest

from areno.api import metrics as metrics_mod
from areno.api.metrics import (
    MetricsRecorder,
    collect_train_batch_stats,
    init_rollout_stats,
    record_rollout_sequence_stats,
)
from areno.api.models import TrainSequence


class MetricsUtilityTest(unittest.TestCase):
    """Metric helper tests cover scalar extraction without TensorBoard writer IO."""

    def test_collect_train_batch_stats_filters_prompt_positions(self):
        """Only response positions should contribute logprob/advantage stats."""
        seq = TrainSequence(
            prompt_mask=[True, True, False, False],
            tokens=[1, 2, 3, 4],
            logprobs=[0.0, 0.0, -0.2, -0.4],
            advantages=[0.0, 0.0, 1.0, -1.0],
            reward=1.0,
        )

        stats = collect_train_batch_stats([seq])

        self.assertEqual(stats["rewards"], [1.0])
        self.assertEqual(stats["logprobs"], [-0.2, -0.4])
        self.assertEqual(stats["advantages"], [1.0, -1.0])
        self.assertEqual(stats["prompt_len"], [2])
        self.assertEqual(stats["response_len"], [2])

    def test_collect_train_batch_stats_keeps_raw_and_transformed_rewards_separate(self):
        """Raw rewards are always collected; transformed rewards only when set."""

        class FakeWriter:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, tag, value, step):
                self.scalars[tag] = value

            def flush(self):
                pass

            def close(self):
                pass

        raw_only = TrainSequence(prompt_mask=[True, False], tokens=[1, 2], reward=2.0)
        shaped = TrainSequence(
            prompt_mask=[True, False], tokens=[1, 2], reward=2.0, transformed_reward=0.5
        )

        disabled_stats = collect_train_batch_stats([raw_only])
        self.assertEqual(disabled_stats["rewards"], [2.0])
        self.assertEqual(disabled_stats["transformed_rewards"], [])

        enabled_stats = collect_train_batch_stats([shaped])
        self.assertEqual(enabled_stats["rewards"], [2.0])
        self.assertEqual(enabled_stats["transformed_rewards"], [0.5])

        writer = FakeWriter()
        from areno.api.metrics import record_training_stats

        record_training_stats(writer, enabled_stats, step=0, train_res={}, train_batch=[shaped])
        # Raw and transformed distributions land under separate tags.
        self.assertIn("rollout/rewards_mean", writer.scalars)
        self.assertIn("rollout/transformed_reward_mean", writer.scalars)
        self.assertIn("rollout/transformed_reward_std", writer.scalars)

        # Disabled batches add no transformed-reward scalars (historic layout).
        writer2 = FakeWriter()
        record_training_stats(writer2, disabled_stats, step=0, train_res={}, train_batch=[raw_only])
        self.assertNotIn("rollout/transformed_reward_mean", writer2.scalars)

    def test_rollout_stats_accumulator_keeps_skip_counters(self):
        """The mutable stats accumulator carries prompt-skip counters forward."""
        stats = init_rollout_stats(skipped_long=2, total_skipped_long=5)

        record_rollout_sequence_stats(stats, prefix_len=3, response_logprobs=[-1.0], response_len=1)

        self.assertEqual(stats["skipped_long"], 2)
        self.assertEqual(stats["total_skipped_long"], 5)
        self.assertEqual(stats["seq_len"], [4])
        self.assertEqual(stats["logprobs"], [-1.0])

    def test_metrics_recorder_close_is_idempotent_context_cleanup(self):
        """MetricsRecorder should close the writer exactly once."""

        class FakeWriter:
            def __init__(self):
                self.close_count = 0

            def close(self):
                self.close_count += 1

        writer = FakeWriter()
        old_factory = metrics_mod.create_tensorboard_writer
        metrics_mod.create_tensorboard_writer = lambda _log_dir: writer
        try:
            with MetricsRecorder("/tmp/areno-test") as recorder:
                self.assertIs(recorder._writer, writer)
            recorder.close()
        finally:
            metrics_mod.create_tensorboard_writer = old_factory

        self.assertEqual(writer.close_count, 1)


if __name__ == "__main__":
    unittest.main()
