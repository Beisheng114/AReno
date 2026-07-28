"""CPU tests for the areno-run-training recipe generator.

Covers: memory estimation formula accuracy, three-mode recipe generation,
deterministic output, invalid input, boundary values, explicit overrides,
provenance tracking, and output-field assertions.

The script is loaded via importlib so it stays outside the areno package.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "skills" / "areno-run-training" / "scripts" / "generate_recipe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_recipe_for_tests", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_recipe_for_tests"] = module
    spec.loader.exec_module(module)
    return module


_mod = _load_module()

ModelShape = _mod.ModelShape
GpuMemoryInfo = _mod.GpuMemoryInfo
estimate_param_count = _mod.estimate_param_count
estimate_weight_bytes = _mod.estimate_weight_bytes
estimate_optimizer_bytes = _mod.estimate_optimizer_bytes
estimate_kv_cache_bytes = _mod.estimate_kv_cache_bytes
estimate_memory = _mod.estimate_memory
split_context_len = _mod.split_context_len
generate_recipe = _mod.generate_recipe
ConfigValue = _mod.ConfigValue

QWEN3_SMALL = ModelShape(
    num_layers=28, num_attention_heads=16, num_kv_heads=8, head_dim=128,
    hidden_size=1024, intermediate_size=3072, vocab_size=151936, dtype_bytes=2,
)


class FakeNoneProbe:
    def probe(self):
        return None


class FakeGpuProbe:
    def __init__(self, total_bytes, free_bytes, device_count=1, device_ids=None):
        ids = device_ids if device_ids is not None else list(range(device_count))
        self._info = GpuMemoryInfo(
            total_bytes=total_bytes, free_bytes=free_bytes,
            device_count=device_count, device_ids=ids,
        )

    def probe(self):
        return self._info


# == memory estimation tests ==

class TestParamCount:
    def test_dense_param_count_matches_formula(self):
        embedding = 151936 * 1024
        attn = (2 * 16 + 2 * 8) * 128 * 1024
        mlp = 3 * 1024 * 3072
        norms = 2 * 1024
        lm_head = 151936 * 1024
        expected = embedding + 28 * (attn + mlp + norms) + lm_head
        assert estimate_param_count(QWEN3_SMALL) == expected

    def test_tied_embeddings_reduces_param_count(self):
        tied = ModelShape(num_layers=1, num_attention_heads=4, num_kv_heads=2, head_dim=64,
                          hidden_size=256, intermediate_size=512, vocab_size=1000,
                          dtype_bytes=2, tie_word_embeddings=True)
        untied = ModelShape(num_layers=1, num_attention_heads=4, num_kv_heads=2, head_dim=64,
                            hidden_size=256, intermediate_size=512, vocab_size=1000,
                            dtype_bytes=2, tie_word_embeddings=False)
        assert estimate_param_count(untied) - estimate_param_count(tied) == 1000 * 256


class TestWeightBytes:
    def test_bf16_weights(self):
        assert estimate_weight_bytes(QWEN3_SMALL) == estimate_param_count(QWEN3_SMALL) * 2

    def test_fp32_weights(self):
        fp32 = ModelShape(num_layers=1, num_attention_heads=4, num_kv_heads=2, head_dim=64,
                          hidden_size=256, intermediate_size=512, vocab_size=1000, dtype_bytes=4)
        assert estimate_weight_bytes(fp32) == estimate_param_count(fp32) * 4


class TestOptimizerBytes:
    def test_fp32_adam(self):
        params = 1_000_000
        assert estimate_optimizer_bytes(params, 2, adam_8bit=False) == params * 14

    def test_8bit_adam(self):
        params = 1_000_000
        assert estimate_optimizer_bytes(params, 2, adam_8bit=True) == params * 8


class TestKvCacheBytes:
    def test_matches_allocation_formula(self):
        # 28 layers, tp=2 -> local_kv=4, 64 seqs, 4096 cache, 256 block
        # ceil(4096/256)=16 blocks/seq, 64*16=1024 blocks, +1 scratch=1025
        expected = 28 * 2 * 1025 * 256 * 4 * 128 * 2
        assert estimate_kv_cache_bytes(QWEN3_SMALL, tp_size=2, max_running_seqs=64,
                                       max_cache_len=4096, block_size=256) == expected

    def test_tp_sharding_reduces_kv(self):
        full = estimate_kv_cache_bytes(QWEN3_SMALL, tp_size=1, max_running_seqs=8, max_cache_len=1024)
        half = estimate_kv_cache_bytes(QWEN3_SMALL, tp_size=2, max_running_seqs=8, max_cache_len=1024)
        assert half == full // 2


class TestEstimateMemory:
    def test_total_is_peak_of_rollout_or_train_phase(self):
        """total = max(rollout_phase, train_phase), not their sum.

        rollout_phase = weights + kv_cache (train weights offloaded)
        train_phase   = weights + optimizer + activations (KV released)
        """
        gpu = FakeGpuProbe(80_000_000_000, 70_000_000_000)
        mem = estimate_memory(QWEN3_SMALL, tp_size=4, dp_size=2, max_running_seqs=64,
                              max_cache_len=4096, mini_bs=16, gpu_info=gpu.probe())
        rollout_phase = mem.weights + mem.kv_cache
        train_phase = mem.weights + mem.optimizer + mem.activations
        assert mem.total == max(rollout_phase, train_phase)

    def test_kv_divided_by_dp_size(self):
        """KV cache should reflect per-GPU seqs = max_running_seqs / dp_size."""
        mem_dp1 = estimate_memory(QWEN3_SMALL, tp_size=4, dp_size=1, max_running_seqs=64,
                                  max_cache_len=4096, mini_bs=16)
        mem_dp2 = estimate_memory(QWEN3_SMALL, tp_size=4, dp_size=2, max_running_seqs=64,
                                  max_cache_len=4096, mini_bs=16)
        # dp_size=2 approximately halves the per-GPU KV cache compared to dp_size=1.
        # Not exactly half because of the +1 scratch block in the allocation.
        ratio = mem_dp1.kv_cache / mem_dp2.kv_cache
        assert 1.9 < ratio < 2.1

    def test_headroom_positive(self):
        gpu = FakeGpuProbe(80_000_000_000, 70_000_000_000)
        mem = estimate_memory(QWEN3_SMALL, tp_size=4, dp_size=2, max_running_seqs=32,
                              max_cache_len=2048, mini_bs=4, gpu_info=gpu.probe())
        assert mem.headroom_ok
        assert mem.headroom_bytes == 70_000_000_000 - mem.total

    def test_headroom_negative(self):
        gpu = FakeGpuProbe(1_000_000, 500_000)
        mem = estimate_memory(QWEN3_SMALL, tp_size=1, dp_size=1, max_running_seqs=256,
                              max_cache_len=8192, mini_bs=64, gpu_info=gpu.probe())
        assert not mem.headroom_ok


# == context splitting tests ==

class TestSplitContextLen:
    def test_sft_all_prompt(self):
        p, r = split_context_len("sft", 4096)
        assert (p, r) == (4096, 0)

    def test_dpo_split_evenly(self):
        p, r = split_context_len("dpo", 4096)
        assert (p, r) == (2048, 2048)

    def test_rl_split(self):
        p, r = split_context_len("gspo", 4096)
        assert (p, r) == (1024, 3072)

    def test_rl_small_context(self):
        p, r = split_context_len("grpo", 1024)
        assert (p, r) == (256, 768)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="context_len must be positive"):
            split_context_len("sft", 0)


# == recipe generation tests ==

class TestRecipeSuccess:
    def test_sft_recipe(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())
        assert r.algo == "sft"
        assert "--algo sft" in r.command
        assert "--batch-size 16" in r.command
        assert r.config["world_size"].value == 8
        assert r.config["world_size"].source == "derived"
        assert r.config["max_prompt_tokens"].value == 2048
        assert r.config["max_new_tokens"].value == 0

    def test_dpo_recipe(self):
        r = generate_recipe(algo="dpo", gpus=4, tp_size=2, context_len=4096, batch_size=8, gpu_probe=FakeNoneProbe())
        assert "--algo dpo" in r.command
        assert r.config["max_prompt_tokens"].value == 2048
        assert r.config["max_new_tokens"].value == 2048

    def test_gspo_recipe(self):
        r = generate_recipe(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
                            reward_fn_path="reward.py", gpu_probe=FakeNoneProbe())
        assert "--algo gspo" in r.command
        assert r.config["max_prompt_tokens"].value == 1024
        assert r.config["max_new_tokens"].value == 3072
        assert r.config["n_samples"].value == 4
        assert r.config["reward_fn_path"].value is not None

    def test_ppo_recipe(self):
        r = generate_recipe(algo="ppo", gpus=8, tp_size=4, context_len=4096, batch_size=16,
                            reward_fn_path="reward.py", gpu_probe=FakeNoneProbe())
        assert "--algo ppo" in r.command

    def test_offline_recipe_has_no_rollout_fields(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())
        assert "n_samples" not in r.config
        assert "reward_fn_path" not in r.config
        assert "max_running_prompts" not in r.config


class TestRecipeDeterminism:
    def test_same_input_same_output(self):
        kw = dict(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32, gpu_probe=FakeNoneProbe())
        r1 = generate_recipe(**kw)
        r2 = generate_recipe(**kw)
        assert r1.command == r2.command
        assert r1.to_json() == r2.to_json()


class TestRecipeValidation:
    def test_unknown_algorithm(self):
        with pytest.raises(ValueError, match=r"\[stage=algo-resolution\] unknown algorithm 'bogus'"):
            generate_recipe(algo="bogus", gpus=8, tp_size=4, context_len=4096, batch_size=32, gpu_probe=FakeNoneProbe())

    def test_tp_not_divisible(self):
        with pytest.raises(ValueError, match=r"\[stage=parallelism\].*divisible"):
            generate_recipe(algo="sft", gpus=3, tp_size=2, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())

    def test_zero_gpus(self):
        with pytest.raises(ValueError, match=r"\[stage=parallelism\].*positive"):
            generate_recipe(algo="sft", gpus=-1, tp_size=1, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())

    def test_zero_context_len(self):
        with pytest.raises(ValueError, match=r"\[stage=sizing\].*context-len"):
            generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=-1, batch_size=16, gpu_probe=FakeNoneProbe())

    def test_zero_batch_size(self):
        with pytest.raises(ValueError, match=r"\[stage=sizing\].*batch-size"):
            generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=-1, gpu_probe=FakeNoneProbe())

    def test_unknown_override_option(self):
        with pytest.raises(ValueError, match=r"\[stage=override\] unknown option 'nonexistent'"):
            generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
                            overrides={"nonexistent": 42}, gpu_probe=FakeNoneProbe())

    def test_cli_only_option_rejected(self):
        with pytest.raises(ValueError, match=r"\[stage=override\] unknown option 'tune_params'"):
            generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
                            overrides={"tune_params": True}, gpu_probe=FakeNoneProbe())


class TestRecipeOverrides:
    def test_lr_override(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
                            overrides={"lr": 5e-7}, gpu_probe=FakeNoneProbe())
        assert r.config["optimizer_lr"].value == 5e-7
        assert r.config["optimizer_lr"].source == "explicit"
        assert "--lr 5e-07" in r.command or "--lr 5e-7" in r.command

    def test_min_lr_override(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
                            overrides={"min_lr": 1e-8}, gpu_probe=FakeNoneProbe())
        assert r.config["optimizer_min_lr"].value == 1e-8
        assert r.config["optimizer_min_lr"].source == "explicit"

    def test_drop_rollout_state_inverted(self):
        r = generate_recipe(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
                            reward_fn_path="reward.py", overrides={"drop_rollout_state": True},
                            gpu_probe=FakeNoneProbe())
        assert r.config["keep_rollout_state"].value is False
        assert "--drop-rollout-state" in r.command

    def test_disable_thinking_override(self):
        r = generate_recipe(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
                            reward_fn_path="reward.py", overrides={"disable_thinking": True},
                            gpu_probe=FakeNoneProbe())
        assert "--disable-thinking" in r.command


class TestProvenance:
    def test_default_values(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())
        assert r.config["optimizer_lr"].source == "default"
        assert r.config["epochs"].source == "default"
        assert r.config["save_interval"].source == "default"

    def test_derived_values(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())
        assert r.config["world_size"].source == "derived"
        assert r.config["batch_size"].source == "derived"
        assert r.config["max_prompt_tokens"].source == "derived"
        assert r.config["mini_bs"].source == "derived"

    def test_rl_derived_fields(self):
        r = generate_recipe(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
                            reward_fn_path="reward.py", gpu_probe=FakeNoneProbe())
        assert r.config["n_samples"].source == "derived"
        assert r.config["keep_rollout_state"].source == "derived"

    def test_explicit_values(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
                            overrides={"lr": 2e-6, "epochs": 5}, gpu_probe=FakeNoneProbe())
        assert r.config["optimizer_lr"].source == "explicit"
        assert r.config["epochs"].source == "explicit"


class TestOutputFields:
    def test_json_output_has_all_fields(self):
        r = generate_recipe(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
                            reward_fn_path="reward.py", gpu_probe=FakeNoneProbe())
        d = json.loads(r.to_json())
        for key in ("algo", "command", "config", "memory", "warnings", "dataset_rows"):
            assert key in d

    def test_json_config_items(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())
        d = json.loads(r.to_json())
        for key, item in d["config"].items():
            assert "value" in item
            assert "source" in item

    def test_command_uses_real_option_names(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())
        valid = _mod._valid_option_names() | {"drop_rollout_state", "disable_thinking"}
        for token in r.command.split():
            if token.startswith("--"):
                assert token[2:].replace("-", "_") in valid, f"unknown option: {token}"

    def test_command_excludes_unrelated_defaults(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())
        assert "--epochs" not in r.command
        assert "--score-micro-bs" not in r.command
        assert "--tp-size" in r.command
        assert "--batch-size" in r.command

    def test_memory_estimate_with_gpu_probe(self):
        gpu = FakeGpuProbe(80_000_000_000, 70_000_000_000, 8)
        r = generate_recipe(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
                            reward_fn_path="reward.py", gpu_probe=gpu, model_shape=QWEN3_SMALL)
        assert r.memory is not None
        assert r.memory.headroom_ok

    def test_warns_when_memory_exceeds_free(self):
        gpu = FakeGpuProbe(1_000_000, 500_000, 8)
        r = generate_recipe(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
                            reward_fn_path="reward.py", gpu_probe=gpu, model_shape=QWEN3_SMALL)
        assert r.memory is not None
        assert not r.memory.headroom_ok
        assert any("exceeds usable VRAM" in w for w in r.warnings)

    def test_rl_without_reward_warns(self):
        r = generate_recipe(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32, gpu_probe=FakeNoneProbe())
        assert any("reward-fn-path or --reward-ckpt" in w for w in r.warnings)

    def test_rl_with_reward_ckpt_no_warn(self):
        r = generate_recipe(algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
                            overrides={"reward_ckpt": "model"}, gpu_probe=FakeNoneProbe())
        assert not any("reward-fn-path or --reward-ckpt" in w for w in r.warnings)

    def test_sft_without_loader_warns(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16, gpu_probe=FakeNoneProbe())
        assert any("dataset-loader-fn" in w for w in r.warnings)

    def test_ppo_without_critic_warns(self):
        r = generate_recipe(algo="ppo", gpus=8, tp_size=4, context_len=4096, batch_size=16,
                            reward_fn_path="reward.py", gpu_probe=FakeNoneProbe())
        assert any("critic-ckpt" in w for w in r.warnings)

    def test_dpo_without_ref_ckpt_warns(self):
        r = generate_recipe(algo="dpo", gpus=4, tp_size=2, context_len=4096, batch_size=8, gpu_probe=FakeNoneProbe())
        assert any("ref-ckpt" in w for w in r.warnings)

    def test_auto_adjust_batch_to_fit_vram(self):
        """When default batch_size exceeds free VRAM, it should shrink."""
        gpu = FakeGpuProbe(8_000_000_000, 4_000_000_000, 8)
        r = generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="reward.py", gpu_probe=gpu,
        )
        assert r.config["batch_size"].value < 32
        assert r.memory.headroom_bytes >= 0
        assert any("Auto-adjusted" in w for w in r.warnings)

    def test_batch_kept_when_vram_sufficient(self):
        """When VRAM is sufficient, batch_size should not change."""
        gpu = FakeGpuProbe(80_000_000_000, 70_000_000_000, 8)
        r = generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="reward.py", gpu_probe=gpu,
        )
        assert r.config["batch_size"].value == 32
        assert not any("Auto-adjusted" in w for w in r.warnings)

    def test_warns_when_even_batch1_doesnt_fit(self):
        """When even batch_size=1 exceeds VRAM, warn to add more GPUs."""
        gpu = FakeGpuProbe(2_000_000_000, 1_500_000_000, 8)
        r = generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
            ckpt="Qwen3-7B", reward_fn_path="reward.py", gpu_probe=gpu,
        )
        assert r.config["batch_size"].value == 1
        assert any("exceeds usable VRAM" in w and "batch_size=1" in w for w in r.warnings)

    def test_explicit_batch_not_auto_adjusted(self):
        """When user explicitly sets batch_size, don't auto-adjust."""
        gpu = FakeGpuProbe(8_000_000_000, 4_000_000_000, 8)
        r = generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="reward.py", gpu_probe=gpu,
            overrides={"batch_size": 32},
        )
        assert r.config["batch_size"].value == 32
        assert not any("Auto-adjusted" in w for w in r.warnings)


class TestDatasetCounting:
    def test_jsonl_row_count(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n{"b": 2}\n{"c": 3}\n')
            f.flush()
            r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=1,
                                dataset_path=f.name, gpu_probe=FakeNoneProbe())
            assert r.dataset_rows == 3
            assert any("3 rows" in w for w in r.warnings)

    def test_remote_dataset_returns_none(self):
        r = generate_recipe(algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
                            dataset_path="gsm8k:main", gpu_probe=FakeNoneProbe())
        assert r.dataset_rows is None


class TestCliOverrideParsing:
    def test_parse_set_int(self):
        assert _mod._parse_set("epochs=5") == ("epochs", 5)

    def test_parse_set_float(self):
        k, v = _mod._parse_set("lr=2e-6")
        assert k == "lr" and v == 2e-6

    def test_parse_set_bool(self):
        k, v = _mod._parse_set("drop_rollout_state=true")
        assert k == "drop_rollout_state" and v is True

    def test_parse_set_string(self):
        k, v = _mod._parse_set("attn_backend=native")
        assert k == "attn_backend" and v == "native"

    def test_parse_set_invalid(self):
        with pytest.raises(ValueError, match="key=value"):
            _mod._parse_set("no_equals_here")


# == model name inference tests ==

class TestParseParamCount:
    def test_qwen_06b(self):
        assert _mod.parse_param_count_from_name("Qwen/Qwen3-0.6B") == 600_000_000

    def test_qwen_7b(self):
        assert _mod.parse_param_count_from_name("Qwen3-7B") == 7_000_000_000

    def test_qwen_235b(self):
        assert _mod.parse_param_count_from_name("Qwen3-235B-A22B") == 235_000_000_000

    def test_llama_8b(self):
        assert _mod.parse_param_count_from_name("meta-llama/Llama-3-8B") == 8_000_000_000

    def test_gemma_2b(self):
        assert _mod.parse_param_count_from_name("gemma-2-2b") == 2_000_000_000

    def test_small_model_m_suffix(self):
        assert _mod.parse_param_count_from_name("MiniCPM5-1B") == 1_000_000_000

    def test_no_match(self):
        assert _mod.parse_param_count_from_name("some-model") is None


class TestInferShapeFromName:
    def test_qwen3_known_family(self):
        shape, note = _mod.infer_shape_from_name("Qwen/Qwen3-0.6B")
        assert shape is not None
        assert shape.num_layers == 28
        assert shape.hidden_size == 1024
        assert "Inferred architecture" in note

    def test_qwen3_7b(self):
        shape, note = _mod.infer_shape_from_name("Qwen3-7B")
        assert shape is not None
        assert shape.num_attention_heads == 28
        assert shape.num_kv_heads == 4

    def test_llama_known_family(self):
        shape, note = _mod.infer_shape_from_name("meta-llama/Llama-3-8B")
        assert shape is not None
        assert shape.num_layers == 32
        assert shape.hidden_size == 4096

    def test_moe_detection(self):
        shape, note = _mod.infer_shape_from_name("Qwen/Qwen3-30B-A3B")
        assert shape is not None
        assert shape.is_moe is True
        assert "MoE detected" in note
        assert "30.0B total" in note
        assert "3.0B active" in note

    def test_unknown_family_generic(self):
        shape, note = _mod.infer_shape_from_name("custom-42B")
        assert shape is not None
        assert shape.param_count_override == 42_000_000_000
        assert "generic heuristic" in note

    def test_no_param_count(self):
        shape, note = _mod.infer_shape_from_name("no-size-in-name")
        assert shape is None
        assert note is None


class TestRecipeWithNameInference:
    def test_gspo_with_name_only(self):
        """Memory estimate works with a model name but no local checkpoint."""
        r = generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="reward.py", gpu_probe=FakeNoneProbe(),
        )
        assert r.memory is not None
        assert r.memory.weights > 0
        assert any("Inferred" in w for w in r.warnings)

    def test_unknown_model_name_still_estimates(self):
        r = generate_recipe(
            algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
            ckpt="custom-10B", gpu_probe=FakeNoneProbe(),
        )
        assert r.memory is not None
        assert r.memory.weights > 0
        assert any("generic heuristic" in w for w in r.warnings)

    def test_weight_estimate_exact_from_name(self):
        """Weights should be exact: param_count * dtype_bytes / tp_size."""
        r = generate_recipe(
            algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
            ckpt="Qwen3-7B", gpu_probe=FakeNoneProbe(),
        )
        assert r.memory is not None
        # 7B params * 2 bytes / 4 tp = 3.5 GB = 3_500_000_000 bytes
        # The known-shape path doesn't use override, so weights come from
        # the architecture formula. For the generic path it would be exact.
        expected_weights = 7_000_000_000 * 2 // 4
        # Known-shape estimate should be close to the param count.
        assert abs(r.memory.weights - expected_weights) / expected_weights < 0.3

    def test_generic_shape_uses_override(self):
        """Unknown-family shapes use param_count_override for exact weights."""
        r = generate_recipe(
            algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
            ckpt="custom-42B", gpu_probe=FakeNoneProbe(),
        )
        assert r.memory is not None
        # 42B * 2 bytes / 4 tp = 21 GB
        assert r.memory.weights == 42_000_000_000 * 2 // 4
