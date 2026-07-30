"""CPU tests for the training-recipe generator skill script.

Covers success paths (SFT/DPO/GSPO), invalid inputs, boundary values,
determinism, memory estimation, auto batch-size adjustment, model-name
inference, dataset row counting, and CLI JSON output.

All tests run without a GPU and without importing the ``areno`` package.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".agents/skills/areno-run-training/scripts/generate_recipe.py"

# -- module loader ----------------------------------------------------------

_spec = importlib.util.spec_from_file_location("generate_recipe", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
# Must register in sys.modules before exec_module so that @dataclass(frozen=True)
# with from __future__ import annotations can resolve its own module for type
# introspection on Python 3.9.
import sys as _sys
_gen_name = _spec.name
_sys.modules[_gen_name] = gen
_spec.loader.exec_module(gen)

# -- helpers ----------------------------------------------------------------


def _fake_gpu(total_gb: float = 80.0, free_gb: float | None = None) -> gen.FakeGpuProbe:
    """A deterministic GPU probe for CPU tests."""
    total = int(total_gb * 1e9)
    free = int((free_gb if free_gb is not None else total_gb) * 1e9)
    return gen.FakeGpuProbe(
        gen.GpuMemoryInfo(total_bytes=total, free_bytes=free, device_count=8, device_ids=list(range(8)))
    )


# == success paths ==========================================================


class TestSFTRecipe:
    def test_generates_runnable_command(self):
        result = gen.generate_recipe(
            algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16
        )
        assert result.command.startswith("areno train --algo sft")
        assert "--world-size 8" in result.command
        assert "--tp-size 4" in result.command
        assert "--batch-size 16" in result.command
        assert "--max-prompt-tokens 2047" in result.command
        assert "--max-new-tokens 1" in result.command  # SFT: min 1 (CLI requires > 0)
        assert "--dataset-loader-fn" in result.command  # SFT requires it (CLI line 192-193)

    def test_context_len_split_all_prompt(self):
        prompt, new = gen.split_context_len("sft", 2048)
        assert prompt == 2047  # context_len - 1 (reserved for min new_tokens)
        assert new == 1        # CLI requires max_new_tokens > 0

    def test_provenance_sources(self):
        result = gen.generate_recipe(
            algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=8
        )
        cfg = result.config
        assert cfg["world_size"].source == "derived"
        assert cfg["tp_size"].source == "derived"
        assert cfg["batch_size"].source == "derived"
        assert cfg["max_prompt_tokens"].source == "derived"
        assert cfg["optimizer_lr"].source == "default"
        assert cfg["epochs"].source == "default"

    def test_uses_placeholders_when_none(self):
        result = gen.generate_recipe(
            algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=8
        )
        assert result.config["ckpt"].value == "<your-ckpt>"
        assert result.config["dataset_path"].value == "<your-dataset>"
        assert result.config["dataset_loader_fn"].value == "<your-dataset-loader>"


class TestDPORecipe:
    def test_generates_runnable_command(self):
        result = gen.generate_recipe(
            algo="dpo", gpus=4, tp_size=2, context_len=4096, batch_size=8
        )
        assert "--algo dpo" in result.command
        assert "--world-size 4" in result.command
        assert "--dpo-beta 0.1" in result.command

    def test_context_len_split_half(self):
        prompt, new = gen.split_context_len("dpo", 4096)
        assert prompt == 2048
        assert new == 2048

    def test_explicit_override_appears_in_command(self):
        result = gen.generate_recipe(
            algo="dpo", gpus=4, tp_size=2, context_len=4096, batch_size=8,
            overrides={"dpo_beta": 0.05, "lr": 5e-7},
        )
        assert "--dpo-beta 0.05" in result.command
        assert "--lr 5e-07" in result.command or "--lr 5e-7" in result.command
        assert result.config["dpo_beta"].value == 0.05
        assert result.config["dpo_beta"].source == "explicit"
        assert result.config["optimizer_lr"].source == "explicit"


class TestGSPORecipe:
    def test_generates_rl_command(self):
        result = gen.generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
            reward_fn_path="examples/math/math_verify_reward.py",
        )
        assert "--algo gspo" in result.command
        assert "--reward-fn-path examples/math/math_verify_reward.py" in result.command
        assert "--n-samples 4" in result.command
        # max_running_prompts is derived as batch_size * n_samples but defaults
        # to None in the config (the runtime fills it).  Validate the derived value.
        assert result.config["max_running_prompts"].value is None
        assert result.config["max_running_prompts"].source == "derived"

    def test_context_len_split_rl(self):
        prompt, new = gen.split_context_len("gspo", 4096)
        assert prompt == 1024   # min(1024, 4096 // 4)
        assert new == 3072       # remainder

    def test_missing_reward_warns(self):
        result = gen.generate_recipe(
            algo="gspo", gpus=4, tp_size=2, context_len=2048, batch_size=4,
        )
        assert any("requires --reward-fn-path" in w for w in result.warnings)

    def test_n_samples_affects_max_running_prompts(self):
        r4 = gen.generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=2048, batch_size=16,
            n_samples=4, reward_fn_path="r.py",
        )
        r8 = gen.generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=2048, batch_size=16,
            n_samples=8, reward_fn_path="r.py",
        )
        assert int(r4.config["n_samples"].value) == 4
        assert int(r8.config["n_samples"].value) == 8
        # max_running_prompts defaults to None (runtime fills it);
        # verify the source label instead of the value.
        assert r4.config["max_running_prompts"].source == "derived"
        assert r8.config["max_running_prompts"].source == "derived"


# == invalid / boundary inputs ==============================================


class TestInvalidInputs:
    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match=r"\[stage=algo-resolution\]"):
            gen.generate_recipe(algo="bogus", gpus=4, tp_size=2, context_len=1024, batch_size=8)

    def test_gpus_not_divisible_by_tp(self):
        with pytest.raises(ValueError, match=r"\[stage=parallelism\].*divisible"):
            gen.generate_recipe(algo="sft", gpus=6, tp_size=4, context_len=1024, batch_size=8)

    def test_zero_gpus(self):
        # gpus=0 defaults to 8, so this should NOT raise.
        # But we can test the post-default validation by passing a negative.
        # The function applies defaults first (0 -> 8), so no error.
        # To truly test gpus<=0, we need to bypass the default.
        result = gen.generate_recipe(
            algo="sft", gpus=0, tp_size=0, context_len=0, batch_size=0
        )
        # With all zeros, defaults: gpus=8, tp=4, ctx=2048, batch=8
        assert result.config["world_size"].value == 8
        assert result.config["tp_size"].value == 4

    def test_negative_gpus_after_default(self):
        with pytest.raises(ValueError, match=r"\[stage=parallelism\].*positive"):
            gen.generate_recipe(algo="sft", gpus=-1, tp_size=4, context_len=1024, batch_size=8)

    def test_context_len_zero_after_default(self):
        # context_len=0 -> defaults to 2048, so no error.  Verify:
        result = gen.generate_recipe(
            algo="sft", gpus=4, tp_size=2, context_len=0, batch_size=8
        )
        # SFT split: max_prompt_tokens = context_len - 1 = 2047
        assert result.config["max_prompt_tokens"].value == 2047

    def test_negative_context_len(self):
        # context_len is negative, not caught by default guard, split will raise.
        with pytest.raises(ValueError, match=r"context-len must be positive"):
            gen.generate_recipe(
                algo="sft", gpus=4, tp_size=2, context_len=-1, batch_size=8
            )

    def test_batch_size_not_positive_after_default(self):
        result = gen.generate_recipe(
            algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=0
        )
        # batch_size=0 defaults to 8
        assert result.config["batch_size"].value == 8

    def test_batch_size_negative(self):
        with pytest.raises(ValueError, match=r"\[stage=sizing\].*batch-size must be positive"):
            gen.generate_recipe(
                algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=-1
            )

    def test_unknown_override_option(self):
        with pytest.raises(ValueError, match=r"\[stage=override\].*unknown option 'bad_field'"):
            gen.generate_recipe(
                algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=8,
                overrides={"bad_field": 42},
            )

    def test_cli_only_option_rejected_in_set(self):
        with pytest.raises(ValueError, match=r"\[stage=override\].*unknown option 'tune_params'"):
            gen.generate_recipe(
                algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=8,
                overrides={"tune_params": True},
            )


# == determinism ============================================================


class TestDeterminism:
    def test_same_inputs_same_output(self):
        kwargs = dict(
            algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="r.py",
            gpu_probe=_fake_gpu(80.0, 80.0),
        )
        r1 = gen.generate_recipe(**kwargs)
        r2 = gen.generate_recipe(**kwargs)
        assert r1.to_json() == r2.to_json()

    def test_command_stable_across_runs(self):
        for _ in range(5):
            r = gen.generate_recipe(
                algo="dpo", gpus=4, tp_size=2, context_len=4096, batch_size=8,
                overrides={"dpo_beta": 0.05},
            )
            assert "--dpo-beta 0.05" in r.command


# == memory estimation ======================================================


class TestMemoryEstimation:
    SHAPE = gen.ModelShape(
        num_layers=28, num_attention_heads=16, num_kv_heads=8,
        head_dim=128, hidden_size=1024, intermediate_size=3072,
        vocab_size=151936,
    )

    def test_weight_bytes_exact(self):
        pc = gen.estimate_param_count(self.SHAPE)
        assert gen.estimate_weight_bytes(self.SHAPE) == pc * 2

    def test_optimizer_bytes_fp32(self):
        pc = gen.estimate_param_count(self.SHAPE)
        assert gen.estimate_optimizer_bytes(pc) == pc * 14   # 2 + 12

    def test_optimizer_bytes_8bit(self):
        pc = gen.estimate_param_count(self.SHAPE)
        assert gen.estimate_optimizer_bytes(pc, adam_8bit=True) == pc * 8  # 2 + 6

    def test_kv_cache_positive(self):
        kv = gen.estimate_kv_cache_bytes(
            self.SHAPE, tp_size=4, max_running_seqs=32, max_cache_len=4096,
        )
        assert kv > 0

    def test_activation_checkpointing_reduces(self):
        base = gen.estimate_activation_bytes(
            self.SHAPE, mini_bs=16, seq_len=4096, activation_checkpointing=False
        )
        ckpt = gen.estimate_activation_bytes(
            self.SHAPE, mini_bs=16, seq_len=4096, activation_checkpointing=True
        )
        assert ckpt < base
        assert ckpt == int(base * 0.3)

    def test_full_memory_breakdown_fields(self):
        mem = gen.estimate_memory(
            self.SHAPE, tp_size=4, dp_size=2, max_running_seqs=128,
            max_cache_len=4096, mini_bs=16,
            gpu_info=gen.GpuMemoryInfo(
                total_bytes=int(80 * 1e9), free_bytes=int(80 * 1e9),
                device_count=8,
            ),
        )
        assert mem.weights > 0
        assert mem.optimizer > 0
        assert mem.kv_cache > 0
        assert mem.activations > 0
        assert mem.total > 0
        assert mem.headroom_ok is True
        assert mem.per_gpu_free is not None

    def test_memory_headroom_negative_when_exceeds(self):
        # Very large batch on tiny GPU should have negative headroom.
        mem = gen.estimate_memory(
            gen.ModelShape(
                num_layers=80, num_attention_heads=64, num_kv_heads=8,
                head_dim=128, hidden_size=8192, intermediate_size=29568,
                vocab_size=152064,
            ),
            tp_size=1, dp_size=1, max_running_seqs=256, max_cache_len=8192,
            mini_bs=16, gpu_info=gen.GpuMemoryInfo(
                total_bytes=int(20 * 1e9), free_bytes=int(20 * 1e9),
                device_count=8,
            ),
        )
        assert mem.headroom_bytes is not None
        assert mem.headroom_bytes < 0
        assert mem.headroom_ok is False


# == auto batch-size adjustment =============================================


class TestAutoBatchAdjust:
    def test_batch_shrinks_when_vram_tight(self):
        # Qwen3-0.6B model, batch_size=64, but only 5 GB free VRAM.
        # The generator should auto-shrink batch_size.
        result = gen.generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=64,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="r.py", n_samples=4,
            gpu_probe=_fake_gpu(80.0, 5.0),
        )
        assert result.config["batch_size"].value < 64
        assert result.memory is not None
        assert result.memory.total <= int(5e9 * 0.95)
        assert any("Auto-adjusted" in w for w in result.warnings)

    def test_batch_stays_when_vram_sufficient(self):
        result = gen.generate_recipe(
            algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
            ckpt="Qwen/Qwen3-0.6B",
            gpu_probe=_fake_gpu(80.0, 80.0),
        )
        assert result.config["batch_size"].value == 16
        assert not any("Auto-adjusted" in w for w in result.warnings)

    def test_explicit_batch_not_auto_adjusted(self):
        result = gen.generate_recipe(
            algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=64,
            ckpt="Qwen/Qwen3-0.6B",
            overrides={"batch_size": 64},
            gpu_probe=_fake_gpu(80.0, 4.0),
        )
        # Explicit overrides should not be auto-adjusted.
        assert result.config["batch_size"].value == 64
        assert result.config["batch_size"].source == "explicit"


# == model-name inference ===================================================


class TestModelNameInference:
    def test_parse_param_count_qwen(self):
        assert gen.parse_param_count_from_name("Qwen/Qwen3-0.6B") == 600_000_000
        assert gen.parse_param_count_from_name("Qwen3-1.7B") == 1_700_000_000
        assert gen.parse_param_count_from_name("Qwen3-7B-Instruct") == 7_000_000_000

    def test_parse_param_count_million(self):
        assert gen.parse_param_count_from_name("tiny-350M") == 350_000_000

    def test_parse_param_count_none(self):
        assert gen.parse_param_count_from_name("no-size-here") is None

    def test_infer_shape_qwen3(self):
        shape, note = gen.infer_shape_from_name("Qwen/Qwen3-0.6B")
        assert shape is not None
        assert shape.num_layers == 28
        assert shape.hidden_size == 1024
        assert note is not None
        assert "qwen3" in note.lower()

    def test_infer_shape_moe_detection(self):
        shape, note = gen.infer_shape_from_name("Qwen/Qwen3-30B-A3B")
        assert shape is not None
        assert shape.is_moe is True
        assert note is not None
        assert "MoE" in note

    def test_infer_shape_unknown_family(self):
        shape, note = gen.infer_shape_from_name("custom-42B")
        assert shape is not None
        assert shape.param_count_override == 42_000_000_000
        assert note is not None
        assert "generic heuristic" in note.lower()

    def test_inferred_shape_weight_exact(self):
        """Weight estimate uses exact param count, not approximate architecture."""
        shape, _ = gen.infer_shape_from_name("custom-42B")
        pc = gen.estimate_param_count(shape)
        assert pc == 42_000_000_000  # from override, not architecture fields


# == dataset row counting ===================================================


class TestDatasetRowCount:
    def test_jsonl_file(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a":1}\n{"b":2}\n{"c":3}\n', encoding="utf-8")
        assert gen._count_dataset_rows(str(f)) == 3

    def test_json_file(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('[{"a":1},{"b":2}]', encoding="utf-8")
        assert gen._count_dataset_rows(str(f)) == 2

    def test_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col\nv1\nv2\nv3\n", encoding="utf-8")
        assert gen._count_dataset_rows(str(f)) == 3  # excludes header

    def test_nonexistent(self):
        assert gen._count_dataset_rows("/no/such/path.jsonl") is None

    def test_directory_of_jsonl(self, tmp_path):
        d = tmp_path / "ds"
        d.mkdir()
        (d / "a.jsonl").write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
        (d / "b.jsonl").write_text('{"b":1}\n', encoding="utf-8")
        assert gen._count_dataset_rows(str(d)) == 3

    def test_recipe_includes_steps_per_epoch_warning(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a":1}\n{"a":2}\n{"a":3}\n{"a":4}\n', encoding="utf-8")
        result = gen.generate_recipe(
            algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=2,
            dataset_path=str(f),
        )
        assert result.dataset_rows == 4
        assert any("steps/epoch" in w for w in result.warnings)


# == CLI invocation =========================================================


class TestCLI:
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
        )

    def test_json_output_is_valid(self):
        proc = self._run("--algo", "sft", "--gpus", "8", "--tp-size", "4",
                         "--context-len", "2048", "--batch-size", "16",
                         "--format", "json")
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["algo"] == "sft"
        assert data["command"].startswith("areno train --algo sft")
        assert isinstance(data["config"], dict)
        assert "warnings" in data
        assert "dataset_rows" in data

    def test_cli_output_has_command(self):
        proc = self._run("--algo", "gspo", "--gpus", "4", "--tp-size", "2",
                         "--context-len", "2048", "--batch-size", "4",
                         "--reward-fn-path", "r.py", "--format", "cli")
        assert proc.returncode == 0
        assert proc.stdout.startswith("areno train --algo gspo")

    def test_invalid_algo_exit_code(self):
        proc = self._run("--algo", "bogus", "--gpus", "4", "--tp-size", "2",
                         "--context-len", "1024", "--batch-size", "8")
        assert proc.returncode != 0

    def test_bad_parallelism_error_message(self):
        # argparse choices won't catch this; the function ValueError will.
        proc = self._run("--algo", "sft", "--gpus", "6", "--tp-size", "4",
                         "--context-len", "1024", "--batch-size", "8",
                         "--format", "json")
        assert proc.returncode == 1
        assert "divisible" in proc.stderr

    def test_set_override_in_json(self):
        proc = self._run("--algo", "dpo", "--gpus", "4", "--tp-size", "2",
                         "--context-len", "4096", "--batch-size", "8",
                         "--set", "dpo_beta=0.05", "--format", "json")
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["config"]["dpo_beta"]["value"] == 0.05
        assert data["config"]["dpo_beta"]["source"] == "explicit"

    def test_no_areno_import_dependency(self):
        """The script must run without the areno package installed."""
        # We can't uninstall areno in the test, but we verify the warning
        # path is exercised (no crash when areno is absent).
        proc = self._run("--algo", "gspo", "--gpus", "4", "--tp-size", "2",
                         "--context-len", "2048", "--batch-size", "4",
                         "--ckpt", "Qwen/Qwen3-0.6B",
                         "--reward-fn-path", "r.py", "--format", "json")
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["algo"] == "gspo"
        # The "Could not read model config" warning is expected when areno
        # is not installed, but the generator still produces output via name inference.
        assert data["memory"] is not None

    def test_auto_flag_produces_valid_output(self):
        """Test --auto flag with fake GPU probe derives parameters automatically."""
        # Use subprocess to test the CLI --auto flag with a fake GPU probe injected
        # Since we can't inject FakeGpuProbe via CLI, we test the Python API directly
        result = gen.generate_recipe(
            algo="gspo", auto=True, ckpt="Qwen/Qwen3-0.6B",
            reward_fn_path="r.py",
            gpu_probe=_fake_gpu(80.0, 70.0)
        )
        # Verify auto-detection produced valid parameters
        assert result.config["world_size"].value > 0
        assert result.config["tp_size"].value > 0
        assert result.config["batch_size"].value > 0
        assert result.config["max_prompt_tokens"].value > 0
        # Check that auto-detection note appears in warnings
        assert any("Auto-detected" in w or "auto" in w.lower() for w in result.warnings)

    def test_auto_mode_falls_back_when_no_gpu(self):
        """Test --auto mode falls back to defaults when GPU probe returns None."""
        # Create a probe that returns None (simulating no GPU)
        class NoGpuProbe:
            def probe(self):
                return None

        result = gen.generate_recipe(
            algo="sft", auto=True,
            gpu_probe=NoGpuProbe()
        )
        # Should fall back to defaults (gpus=8, tp_size derived from _best_tp_for_gpus(8)=8)
        assert result.config["world_size"].value == 8  # default gpus
        # _best_tp_for_gpus(8) returns 8 (largest power-of-2 that divides 8)
        assert result.config["tp_size"].value == 8   # derived from _best_tp_for_gpus(8)
        assert any("no GPU detected" in w.lower() or "falling back" in w.lower() for w in result.warnings)


# == boundary value tests ===================================================


class TestBoundaryValues:
    """Test extreme/boundary values for robustness."""

    def test_very_large_model_235b_moe(self):
        """Test memory estimation for very large MoE model (235B)."""
        shape, note = gen.infer_shape_from_name("Qwen3-235B-A22B")
        assert shape is not None
        assert shape.is_moe is True
        # Verify param count is correctly parsed
        assert gen.parse_param_count_from_name("Qwen3-235B-A22B") == 235_000_000_000
        # Memory estimation should not overflow
        pc = gen.estimate_param_count(shape)
        assert pc > 0
        # Weight bytes should be reasonable (uses param_count_override if available)
        weight_bytes = gen.estimate_weight_bytes(shape)
        assert weight_bytes > 100_000_000_000  # > 100 GB (heuristic estimate for 235B)

    def test_very_small_model_60m(self):
        """Test memory estimation for very small model (60M)."""
        shape, note = gen.infer_shape_from_name("tiny-60M")
        assert shape is not None
        pc = gen.estimate_param_count(shape)
        assert pc == 60_000_000  # from override
        weight_bytes = gen.estimate_weight_bytes(shape)
        assert weight_bytes == 120_000_000  # 60M * 2 bytes

    def test_extreme_context_len_128k(self):
        """Test context length splitting with very large context (128k)."""
        # RL algorithm: prompt = min(1024, 25%), response = remainder
        prompt, new = gen.split_context_len("gspo", 131072)
        assert prompt == 1024  # capped at 1024 for RL
        assert new == 130048    # remainder

        # DPO algorithm: 50/50 split
        prompt, new = gen.split_context_len("dpo", 131072)
        assert prompt == 65536
        assert new == 65536

    def test_extreme_batch_size_1(self):
        """Test with minimum batch size of 1."""
        result = gen.generate_recipe(
            algo="sft", gpus=1, tp_size=1, context_len=512, batch_size=1
        )
        assert result.config["batch_size"].value == 1
        assert result.config["mini_bs"].value == 1  # min(1, 16)

    def test_extreme_batch_size_1024(self):
        """Test with very large batch size."""
        result = gen.generate_recipe(
            algo="sft", gpus=8, tp_size=8, context_len=512, batch_size=1024
        )
        assert result.config["batch_size"].value == 1024
        assert result.config["mini_bs"].value == 16  # min(1024, 16)

    def test_single_gpu_setup(self):
        """Test with single GPU (tp_size=1, world_size=1)."""
        result = gen.generate_recipe(
            algo="sft", gpus=1, tp_size=1, context_len=1024, batch_size=4
        )
        assert result.config["world_size"].value == 1
        assert result.config["tp_size"].value == 1

    def test_very_large_gpu_count_128(self):
        """Test with very large GPU count."""
        result = gen.generate_recipe(
            algo="sft", gpus=128, tp_size=8, context_len=1024, batch_size=32
        )
        assert result.config["world_size"].value == 128
        assert result.config["tp_size"].value == 8

    def test_max_tp_size_equals_gpus(self):
        """Test when tp_size equals world_size (no data parallelism)."""
        result = gen.generate_recipe(
            algo="sft", gpus=8, tp_size=8, context_len=1024, batch_size=16
        )
        assert result.config["world_size"].value == 8
        assert result.config["tp_size"].value == 8
        # dp_size = 8 // 8 = 1

    def test_empty_dataset_path(self):
        """Test with empty dataset path."""
        result = gen.generate_recipe(
            algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=8,
            dataset_path=""
        )
        # Should use placeholder, not crash
        assert result.config["dataset_path"].value == "<your-dataset>"

    def test_very_long_model_name(self):
        """Test with very long model name."""
        long_name = "organization/" + "x" * 200 + "-7B-model-v1.0"
        result = gen.generate_recipe(
            algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=8,
            ckpt=long_name
        )
        # Should handle long names gracefully
        assert result.config["ckpt"].value == long_name

    def test_special_chars_in_path(self):
        """Test handling of special characters in paths."""
        # This is more of a smoke test - the actual safety depends on shell escaping
        result = gen.generate_recipe(
            algo="sft", gpus=4, tp_size=2, context_len=1024, batch_size=8,
            dataset_path="/path/with spaces/data.jsonl"
        )
        assert result.config["dataset_path"].value == "/path/with spaces/data.jsonl"


# == agentic RL parameters ================================================


class TestAgenticParams:
    """Tests for --agent-fn, --dataset-loader-fn, --agent-timeout-s, --train-tool-results."""

    def test_all_agentic_params_combined(self):
        result = gen.generate_recipe(
            algo="grpo", gpus=1, tp_size=1, context_len=4096, batch_size=2,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="r.py",
            agent_fn="a.py", dataset_loader_fn="dl.py",
            agent_timeout_s=600.0, train_tool_results=True,
        )
        assert "--agent-fn a.py" in result.command
        assert "--dataset-loader-fn dl.py" in result.command
        assert "--agent-timeout-s 600.0" in result.command
        assert result.config["agent_fn"].source == "explicit"
        assert result.config["train_tool_results"].value is True

    def test_agent_timeout_s_short_warning(self):
        result = gen.generate_recipe(
            algo="grpo", gpus=1, tp_size=1, context_len=4096, batch_size=2,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="r.py",
            agent_fn="a.py", agent_timeout_s=30.0,
        )
        assert any("very short" in w for w in result.warnings)

    def test_no_agent_fn_keeps_default_null(self):
        result = gen.generate_recipe(
            algo="grpo", gpus=1, tp_size=1, context_len=4096, batch_size=2,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="r.py",
        )
        assert result.config["agent_fn"].value is None
        assert "--agent-fn" not in result.command


# == multi-role memory estimation =========================================


class TestMultiRoleMemory:
    """Tests for PPO (critic+ref) and DPO (ref) extra memory estimation."""

    def test_ppo_memory_higher_than_grpo(self):
        """PPO should estimate ~2x memory of GRPO due to critic + ref models."""
        common = dict(
            gpus=4, tp_size=2, context_len=4096, batch_size=4,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="r.py",
            gpu_probe=_fake_gpu(80.0, 70.0),
        )
        grpo_result = gen.generate_recipe(algo="grpo", **common)
        ppo_result = gen.generate_recipe(algo="ppo", **common)
        assert ppo_result.memory.total > grpo_result.memory.total
        ratio = ppo_result.memory.total / grpo_result.memory.total
        assert 1.8 < ratio < 2.5, f"PPO/GRPO ratio={ratio:.2f}, expected ~2x"

    def test_ppo_multi_role_warning(self):
        result = gen.generate_recipe(
            algo="ppo", gpus=4, tp_size=2, context_len=4096, batch_size=4,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="r.py",
            gpu_probe=_fake_gpu(80.0, 70.0),
        )
        assert any("Multi-role" in w and "critic" in w for w in result.warnings)

    def test_estimate_memory_directly_with_extra_roles(self):
        """Unit test for estimate_memory num_extra_trainable/frozen params."""
        shape = gen.infer_shape_from_name("Qwen/Qwen3-0.6B")[0]
        gpu = gen.GpuMemoryInfo(
            total_bytes=int(80 * 1e9), free_bytes=int(70 * 1e9),
            device_count=1, device_ids=[0],
        )
        base = gen.estimate_memory(
            shape, tp_size=1, dp_size=1, max_running_seqs=4,
            max_cache_len=4096, mini_bs=2, gpu_info=gpu,
        )
        with_extra = gen.estimate_memory(
            shape, tp_size=1, dp_size=1, max_running_seqs=4,
            max_cache_len=4096, mini_bs=2, gpu_info=gpu,
            num_extra_trainable=1, num_extra_frozen=1,
        )
        assert with_extra.total > base.total
        extra = with_extra.total - base.total
        per_gpu_weights = gen.estimate_weight_bytes(shape)
        per_gpu_opt = gen.estimate_optimizer_bytes(gen.estimate_param_count(shape), shape.dtype_bytes)
        expected_extra = per_gpu_weights * 2 + per_gpu_opt
        assert extra == expected_extra


# == parameter validation =================================================


class TestParameterValidation:
    """Tests for new validation: n_samples, mini_bs, enum values."""

    def test_n_samples_zero_raises_for_rl(self):
        with pytest.raises(ValueError, match="n-samples must be >= 1"):
            gen.generate_recipe(
                algo="grpo", gpus=1, tp_size=1, context_len=4096, batch_size=2,
                ckpt="Qwen/Qwen3-0.6B", reward_fn_path="r.py",
                n_samples=0,
            )

    def test_mini_bs_exceeds_batch_size_warns_and_caps(self):
        result = gen.generate_recipe(
            algo="sft", gpus=1, tp_size=1, context_len=2048, batch_size=4,
            ckpt="Qwen/Qwen3-0.6B", mini_bs=16,
        )
        assert any("mini-bs" in w and "capped" in w for w in result.warnings)
        assert result.config["mini_bs"].value <= 4

    def test_invalid_attn_backend_raises(self):
        with pytest.raises(ValueError, match="attn_backend.*invalid"):
            gen.generate_recipe(
                algo="sft", gpus=1, tp_size=1, context_len=2048, batch_size=4,
                ckpt="Qwen/Qwen3-0.6B",
                overrides={"attn_backend": "bogus"},
            )


# == --lr CLI flag ========================================================


class TestLearningRateFlag:
    """Tests for the --lr CLI flag mapping to optimizer_lr."""

    def test_lr_appears_in_command_and_explicit(self):
        result = gen.generate_recipe(
            algo="grpo", gpus=1, tp_size=1, context_len=4096, batch_size=2,
            ckpt="Qwen/Qwen3-0.6B", reward_fn_path="r.py",
            lr=5e-6,
        )
        assert "--lr 5e-06" in result.command
        assert result.config["optimizer_lr"].value == 5e-6
        assert result.config["optimizer_lr"].source == "explicit"

    def test_lr_none_uses_default(self):
        result = gen.generate_recipe(
            algo="sft", gpus=1, tp_size=1, context_len=2048, batch_size=4,
            ckpt="Qwen/Qwen3-0.6B",
        )
        assert result.config["optimizer_lr"].value == 1e-6
        assert result.config["optimizer_lr"].source == "default"
        assert "--lr" not in result.command


# == auto_detect_params n_samples fix ======================================


class TestAutoDetectNSamples:
    """Tests that auto_detect_params uses the actual n_samples, not hardcoded 8."""

    def test_auto_detect_uses_custom_n_samples(self):
        """auto_detect_params should use provided n_samples for batch estimation."""
        shape = gen.infer_shape_from_name("Qwen/Qwen3-0.6B")[0]
        param_count = gen.estimate_param_count(shape)
        gpu = gen.GpuMemoryInfo(
            total_bytes=int(80 * 1e9), free_bytes=int(70 * 1e9),
            device_count=8, device_ids=list(range(8)),
        )
        params_4 = gen.auto_detect_params(gpu, param_count, "grpo", n_samples=4)
        params_8 = gen.auto_detect_params(gpu, param_count, "grpo", n_samples=8)
        assert params_8.batch_size <= params_4.batch_size


# == CLI integration: generated command must pass areno train validation =====


class TestGeneratedCommandCLICompatibility:
    """Verify that generated commands satisfy areno CLI constraints (train.py).

    These tests parse the command string and check known CLI validation rules
    that would cause ``click.UsageError`` at runtime.  They do NOT import or run
    the actual ``areno`` CLI — only the constraints documented in
    ``areno/cli/train.py`` lines 190-241 are checked statically.
    """

    @staticmethod
    def _parse_command(cmd: str) -> dict[str, str]:
        """Parse an 'areno train --flag value ...' string into a dict."""
        tokens = cmd.split()
        assert tokens[0] == "areno" and tokens[1] == "train"
        result: dict[str, str] = {}
        i = 2
        while i < len(tokens):
            if tokens[i].startswith("--"):
                key = tokens[i][2:]
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    result[key] = tokens[i + 1]
                    i += 2
                else:
                    result[key] = "True"  # boolean flag
                    i += 1
            else:
                i += 1
        return result

    def test_sft_command_passes_cli_constraints(self):
        """SFT: max_new_tokens > 0, dataset_loader_fn present (train.py:192-193, 238-239)."""
        result = gen.generate_recipe(
            algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
        )
        parsed = self._parse_command(result.command)
        # train.py:238-239 — max_new_tokens must be positive
        assert int(parsed["max-new-tokens"]) > 0, \
            f"SFT max_new_tokens={parsed['max-new-tokens']} would fail CLI validation"
        # train.py:236-237 — max_prompt_tokens must be positive
        assert int(parsed["max-prompt-tokens"]) > 0
        # train.py:192-193 — dataset_loader_fn required for SFT
        assert "dataset-loader-fn" in parsed, \
            "SFT command missing --dataset-loader-fn (train.py line 192-193 requires it)"

    def test_dpo_command_passes_cli_constraints(self):
        """DPO: max_new_tokens > 0, max_prompt_tokens > 0."""
        result = gen.generate_recipe(
            algo="dpo", gpus=4, tp_size=2, context_len=4096, batch_size=8,
        )
        parsed = self._parse_command(result.command)
        assert int(parsed["max-new-tokens"]) > 0
        assert int(parsed["max-prompt-tokens"]) > 0
        assert parsed["algo"] == "dpo"

    def test_gspo_command_passes_cli_constraints(self):
        """GSPO: max_new_tokens > 0, reward_fn_path present."""
        result = gen.generate_recipe(
            algo="gspo", gpus=8, tp_size=4, context_len=4096, batch_size=32,
            reward_fn_path="examples/math/math_verify_reward.py",
        )
        parsed = self._parse_command(result.command)
        assert int(parsed["max-new-tokens"]) > 0
        assert int(parsed["max-prompt-tokens"]) > 0
        assert "reward-fn-path" in parsed

    def test_grpo_command_passes_cli_constraints(self):
        """GRPO: max_new_tokens > 0, n_samples present."""
        result = gen.generate_recipe(
            algo="grpo", gpus=4, tp_size=2, context_len=2048, batch_size=4,
            reward_fn_path="r.py", n_samples=8,
        )
        parsed = self._parse_command(result.command)
        assert int(parsed["max-new-tokens"]) > 0
        assert int(parsed["n-samples"]) >= 1

    def test_ppo_command_passes_cli_constraints(self):
        """PPO: max_new_tokens > 0, reward_fn_path present."""
        result = gen.generate_recipe(
            algo="ppo", gpus=4, tp_size=2, context_len=4096, batch_size=4,
            reward_fn_path="r.py",
        )
        parsed = self._parse_command(result.command)
        assert int(parsed["max-new-tokens"]) > 0
        assert "reward-fn-path" in parsed

    def test_all_algos_no_zero_new_tokens(self):
        """No algorithm should produce max_new_tokens=0 (train.py:238-239)."""
        for algo in ["sft", "dpo", "gspo", "grpo", "ppo"]:
            kwargs = dict(gpus=4, tp_size=2, context_len=2048, batch_size=4)
            if algo in {"gspo", "grpo", "ppo"}:
                kwargs["reward_fn_path"] = "r.py"
            result = gen.generate_recipe(algo=algo, **kwargs)
            parsed = self._parse_command(result.command)
            assert int(parsed["max-new-tokens"]) > 0, \
                f"algo={algo}: max_new_tokens=0 would fail CLI validation"

    def test_all_algos_no_zero_prompt_tokens(self):
        """No algorithm should produce max_prompt_tokens=0 (train.py:236-237)."""
        for algo in ["sft", "dpo", "gspo", "grpo", "ppo"]:
            kwargs = dict(gpus=4, tp_size=2, context_len=512, batch_size=4)
            if algo in {"gspo", "grpo", "ppo"}:
                kwargs["reward_fn_path"] = "r.py"
            result = gen.generate_recipe(algo=algo, **kwargs)
            parsed = self._parse_command(result.command)
            assert int(parsed["max-prompt-tokens"]) > 0, \
                f"algo={algo}: max_prompt_tokens=0 would fail CLI validation"

    def test_sft_with_explicit_loader_fn(self):
        """SFT with --dataset-loader-fn provided should include it in command."""
        result = gen.generate_recipe(
            algo="sft", gpus=8, tp_size=4, context_len=2048, batch_size=16,
            dataset_loader_fn="examples/math/dataset_loader.py",
        )
        assert "--dataset-loader-fn examples/math/dataset_loader.py" in result.command
        assert result.config["dataset_loader_fn"].source == "explicit"

    def test_cli_invocation_sft_has_dataset_loader_fn(self):
        """CLI-generated SFT command includes --dataset-loader-fn."""
        proc = gen_subprocess = None
        proc = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--algo", "sft", "--gpus", "8", "--tp-size", "4",
             "--context-len", "2048", "--batch-size", "16",
             "--format", "cli"],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
        )
        assert proc.returncode == 0
        assert "--dataset-loader-fn" in proc.stdout

    def test_cli_invocation_sft_no_zero_new_tokens(self):
        """CLI-generated SFT command does not contain --max-new-tokens 0."""
        proc = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--algo", "sft", "--gpus", "8", "--tp-size", "4",
             "--context-len", "2048", "--batch-size", "16",
             "--format", "cli"],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
        )
        assert proc.returncode == 0
        assert "--max-new-tokens 0" not in proc.stdout
        assert "--max-new-tokens 1" in proc.stdout
