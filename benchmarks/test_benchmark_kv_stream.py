#!/usr/bin/env python3
"""Unit tests for the automatic adaptive KV benchmark driver."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("benchmark_kv_stream.py")
SPEC = importlib.util.spec_from_file_location("benchmark_kv_stream", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkKvStreamTest(unittest.TestCase):
    def test_parse_token_count(self) -> None:
        self.assertEqual(BENCHMARK.parse_token_count("192K"), 192 * 1024)
        self.assertEqual(BENCHMARK.parse_token_count("262144"), 262144)
        with self.assertRaises(argparse.ArgumentTypeError):
            BENCHMARK.parse_token_count("bad")

    def test_parse_args_resolves_launched_paths(self) -> None:
        model = Path("models/model.gguf")
        server = Path("build/bin/llama-server")
        args = BENCHMARK.parse_args(
            [
                "--model", str(model),
                "--server", str(server),
                "--max-context", "8K",
                "--batch-size", "768",
                "--ubatch-size", "512",
            ]
        )
        self.assertEqual(args.model, model.resolve())
        self.assertEqual(args.server, server.resolve())
        self.assertEqual(args.batch_size, 768)
        self.assertEqual(args.ubatch_size, 512)

    def test_validate_args_rejects_ubatch_larger_than_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            server = root / "llama-server"
            model.touch()
            server.touch(mode=0o755)
            args = BENCHMARK.parse_args(
                [
                    "--model", str(model),
                    "--server", str(server),
                    "--max-context", "8K",
                    "--batch-size", "256",
                    "--ubatch-size", "512",
                ]
            )
            with self.assertRaisesRegex(SystemExit, "must not exceed"):
                BENCHMARK.validate_args(args)

    def test_context_capacities_include_non_aligned_maximum(self) -> None:
        self.assertEqual(
            BENCHMARK.context_capacities(20000),
            [8192, 16384, 20000],
        )

    def test_context_capacities_honor_custom_start_and_step(self) -> None:
        self.assertEqual(
            BENCHMARK.context_capacities(163840, 40960, 8192),
            list(range(40960, 163841, 8192)),
        )


    def test_pool_estimate_uses_all_free_memory_and_rounds_down(self) -> None:
        self.assertEqual(BENCHMARK.estimate_pool_mib(64, 3500, 32), 3552)
        self.assertEqual(
            BENCHMARK.estimate_pool_mib(64, 3500, 32, max_pool_mib=2048),
            2048,
        )

    def test_clean_server_env_removes_memory_policy_overrides(self) -> None:
        inherited = {
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
            "GGML_CUDA_PREFER_MODEL_WEIGHTS": "1",
            "GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS": "8",
            "KEEP_ME": "yes",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            env = BENCHMARK.clean_server_env("2")
        self.assertNotIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", env)
        self.assertNotIn("GGML_CUDA_PREFER_MODEL_WEIGHTS", env)
        self.assertNotIn("GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS", env)
        self.assertEqual(env["KEEP_ME"], "yes")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        pinned = BENCHMARK.clean_server_env(None, fixed_ring_slots=380)
        self.assertEqual(pinned["GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS"], "380")
        traced = BENCHMARK.clean_server_env(None, trace_kv_stream=True)
        self.assertEqual(traced["LLAMA_KV_STREAM_TRACE"], "1")

    def test_fixed_ring_slots_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            server = root / "llama-server"
            model.touch()
            server.touch(mode=0o755)
            base = [
                "--model", str(model), "--server", str(server),
                "--max-context", "64K", "--min-context", "8K",
            ]
            args = BENCHMARK.parse_args(base + ["--fixed-ring-slots", "0"])
            self.assertEqual(args.fixed_ring_slots, 0)
            with self.assertRaisesRegex(SystemExit, "fixed ring slots"):
                BENCHMARK.validate_args(args)
            BENCHMARK.validate_args(BENCHMARK.parse_args(base + ["--fixed-ring-slots", "380"]))

    def test_server_command_uses_tested_configuration(self) -> None:
        args = argparse.Namespace(
            server=Path("/tmp/llama-server"),
            model=Path("/tmp/model.gguf"),
            port=12355,
            extra_server_arg=["--verbosity", "3"],
            cache_type_k="bf16",
            cache_type_v="q8_0",
            batch_size=768,
            ubatch_size=512,
            parallel=1,
            kv_unified="auto",
            n_gpu_layers="all",
            kv_stream_cpu_threads=None,
            kv_stream_cpu_share=None,
        )
        command = BENCHMARK.server_command(args, 131072, 2304)
        self.assertEqual(command[0], "/tmp/llama-server")
        self.assertIn("131072", command)
        self.assertIn("2304", command)
        self.assertEqual(command[command.index("-ctk") + 1], "bf16")
        self.assertEqual(command[command.index("-ctv") + 1], "q8_0")
        self.assertEqual(command[command.index("-b") + 1], "768")
        self.assertEqual(command[command.index("-ub") + 1], "512")
        self.assertEqual(command[command.index("-np") + 1], "1")
        self.assertNotIn("--kv-unified", command)
        self.assertNotIn("--no-kv-unified", command)
        self.assertEqual(command[-2:], ["--verbosity", "3"])

    def test_server_command_passes_slots_and_cache_mode(self) -> None:
        args = argparse.Namespace(
            server=Path("/tmp/llama-server"),
            model=Path("/tmp/model.gguf"),
            port=12355,
            extra_server_arg=[],
            cache_type_k="q8_0",
            cache_type_v="q4_0",
            batch_size=256,
            ubatch_size=256,
            parallel=6,
            kv_unified="on",
            n_gpu_layers="20",
            kv_stream_cpu_threads=None,
            kv_stream_cpu_share=None,
        )
        command = BENCHMARK.server_command(args, 262144, 512)
        self.assertEqual(command[command.index("-np") + 1], "6")
        self.assertEqual(command[command.index("-ngl") + 1], "20")
        self.assertIn("--kv-unified", command)
        args.kv_unified = "off"
        self.assertIn("--no-kv-unified", BENCHMARK.server_command(args, 262144, 512))

    def test_request_shapes_defaults_to_the_slot_capacity(self) -> None:
        args = argparse.Namespace(fill=None, kv_unified="auto", parallel=1)
        self.assertEqual(BENCHMARK.request_shapes(args, 262144), [(262144,)])
        args.fill = [[4096], [16384], [102400, 10240]]
        self.assertEqual(
            BENCHMARK.request_shapes(args, 262144),
            [(4096,), (16384,), (102400, 10240)],
        )

    def test_slot_capacity_splits_unless_the_cache_is_unified(self) -> None:
        args = argparse.Namespace(kv_unified="off", parallel=6)
        self.assertEqual(BENCHMARK.slot_capacity(args, 262144), 43690)
        args.kv_unified = "auto"
        self.assertEqual(BENCHMARK.slot_capacity(args, 262144), 43690)
        args.kv_unified = "on"
        self.assertEqual(BENCHMARK.slot_capacity(args, 262144), 262144)
        self.assertEqual(
            BENCHMARK.slot_capacity(
                argparse.Namespace(kv_unified="auto", parallel=1), 262144
            ),
            262144,
        )

    def test_parse_fill_separates_points_from_concurrent_requests(self) -> None:
        self.assertEqual(BENCHMARK.parse_fill("4K,16K"), [[4096], [16384]])
        self.assertEqual(BENCHMARK.parse_fill("100K+10K"), [[102400, 10240]])
        self.assertEqual(
            BENCHMARK.parse_fill("4K,100K+10K"), [[4096], [102400, 10240]]
        )
        self.assertIsNone(BENCHMARK.parse_fill(None))

    def test_validate_rejects_fill_beyond_the_slot_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            server = root / "llama-server"
            model.touch()
            server.touch(mode=0o755)
            base = [
                "--model", str(model), "--server", str(server),
                "--max-context", "256K",
            ]
            args = BENCHMARK.parse_args(
                base + ["--min-context", "256K", "--fill", "100K",
                        "--parallel", "6", "--kv-unified", "off"]
            )
            with self.assertRaisesRegex(SystemExit, "slot capacity"):
                BENCHMARK.validate_args(args)
            args = BENCHMARK.parse_args(
                base + ["--min-context", "8K", "--fill", "100K"]
            )
            with self.assertRaisesRegex(SystemExit, "slot capacity"):
                BENCHMARK.validate_args(args)
            args = BENCHMARK.parse_args(
                base + ["--min-context", "256K", "--fill", "100K+10K"]
            )
            with self.assertRaisesRegex(SystemExit, "needs .* slots"):
                BENCHMARK.validate_args(args)

    def test_summarize_streams_separates_decode_rate_from_end_to_end(self) -> None:
        streams = [
            {"prefill_tps": 100.0, "decode_tps": 20.0, "prompt_ms": 10.0, "predicted_ms": 50.0},
            {"prefill_tps": 200.0, "decode_tps": 30.0, "prompt_ms": 40.0, "predicted_ms": 20.0},
        ]
        summary = BENCHMARK.summarize_streams(streams, window_seconds=10.0, decode_tokens=256)
        self.assertEqual(summary["concurrency"], 2)
        self.assertEqual(summary["decode_tps"], 50.0)
        self.assertEqual(summary["end_to_end_tps"], 51.2)
        self.assertEqual(summary["prefill_tps"], 300.0)
        self.assertEqual(summary["prompt_ms"], 40.0)
        self.assertEqual(summary["predicted_ms"], 50.0)

    def test_trace_parser_marks_only_pages_beyond_resident_partition(self) -> None:
        log = (
            "0.01.000.000 W kv_stream_adapt: active 65536, resident 256, ring 32, layout 256, "
            "samples 1, misses 0, copy busy 0.0%, peak 1, skipped 0, resident attended 0, "
            "cpu pages 0, cpu declines prefill/no eligible/below min 17/0/0\n"
            "0.01.001.000 W kv_stream_adapt: adaptive KV partition: resident pages/layer "
            "256 -> 248, ring slots 32 -> 160, miss 50.0%, copy busy 25.0%\n"
            "0.01.002.000 W kv_stream_adapt: active 65792, resident 248, ring 160, layout 248, "
            "samples 2, misses 1, copy busy 25.0%, peak 10, skipped 4, resident attended 8, "
            "cpu pages 300, cpu declines prefill/no eligible/below min 0/1/0\n"
            "0.01.003.000 W kv_stream_adapt: active 66048, resident 248, ring 160, "
            "samples 0, misses 0, copy busy 25.0%, peak 10, skipped 0, resident attended 2, "
            "cpu pages 140, cpu declines prefill/no eligible/below min 0/0/2\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(log)
            parsed = BENCHMARK.parse_kv_stream_trace(path)
        self.assertTrue(parsed["streaming_active"])
        self.assertEqual(parsed["stream_first_active_tokens"], 65792)
        self.assertEqual(parsed["stream_trace_samples"], 3)
        self.assertEqual(parsed["stream_repartitions"], 1)
        self.assertEqual(parsed["stream_max_ring_slots"], 160)
        self.assertEqual(parsed["stream_zero_sample_windows"], 1)
        self.assertEqual(parsed["stream_total_cpu_pages"], 440)
        self.assertEqual(parsed["stream_max_copy_busy"], 25.0)
        self.assertEqual(parsed["stream_total_cpu_decline_prefill"], 17)
        self.assertEqual(parsed["stream_total_cpu_decline_no_eligible"], 1)
        self.assertEqual(parsed["stream_total_cpu_decline_below_min"], 2)
        busy = [window for window in parsed["stream_windows"] if window["active_tokens"] == 65792]
        self.assertEqual(len(busy), 1)
        self.assertEqual(busy[0]["samples"], 2)
        self.assertEqual(busy[0]["misses"], 1)
        self.assertEqual(busy[0]["peak"], 10)
        self.assertEqual(busy[0]["skipped_pages"], 4)
        self.assertEqual(busy[0]["resident_pages_attended"], 8)
        self.assertEqual(busy[0]["cpu_pages"], 300)
        self.assertEqual(busy[0]["cpu_decline_no_eligible"], 1)


    def test_resume_rejects_changed_settings(self) -> None:

        signature = {"model": "/tmp/model.gguf", "max_context": 16384}
        BENCHMARK.validate_resume(signature.copy(), signature, Path("results.jsonl"))
        with self.assertRaisesRegex(SystemExit, "different settings"):
            BENCHMARK.validate_resume(
                {"model": "/tmp/other.gguf", "max_context": 16384},
                signature,
                Path("results.jsonl"),
            )

    def test_csv_and_plot_accept_partial_sweep(self) -> None:
        rows = {
            8192: {
                "context_capacity": 8192,
                "prompt_tokens": 7936,
                "decode_tokens": 256,
                "pool_mib": 3552,
                "prefill_tps": 1400.0,
                "decode_tps": 50.0,
            },
            16384: {
                "context_capacity": 16384,
                "prompt_tokens": 16128,
                "decode_tokens": 256,
                "pool_mib": 3520,
                "prefill_tps": 1300.0,
                "decode_tps": 45.0,
            },
        }
        try:
            plt = BENCHMARK.require_matplotlib()
        except SystemExit:
            self.skipTest("Matplotlib is not installed")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            BENCHMARK.write_csv(output / "results.csv", rows)
            BENCHMARK.plot_results(output, rows, plt)
            self.assertTrue((output / "results.csv").is_file())
            self.assertTrue((output / "kv-stream-sweep.png").is_file())
            self.assertTrue((output / "kv-stream-sweep.svg").is_file())

    def test_clean_server_env_removes_kv_stream_knobs(self) -> None:
        inherited = {
            "GGML_CUDA_KV_STREAM_CPU_SHARE": "0.6",
            "GGML_CUDA_KV_STREAM_CPU_CPUS": "0-3",
            "GGML_CUDA_KV_STREAM_PARTS": "4",
            "LLAMA_ARG_KV_STREAM_CPU_THREADS": "6",
            "LLAMA_ARG_KV_STREAM_CPU_SHARE": "auto",
            "LLAMA_ARG_CTX_SIZE": "4096",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            env = BENCHMARK.clean_server_env(None)
            pinned = BENCHMARK.clean_server_env(
                None, server_env=["GGML_CUDA_KV_STREAM_CPU_CPUS=2-7"])
        for name in inherited:
            if name != "LLAMA_ARG_CTX_SIZE":
                self.assertNotIn(name, env)
        self.assertEqual(env["LLAMA_ARG_CTX_SIZE"], "4096")
        self.assertEqual(pinned["GGML_CUDA_KV_STREAM_CPU_CPUS"], "2-7")
        self.assertNotIn("GGML_CUDA_KV_STREAM_CPU_SHARE", pinned)

    def test_server_command_passes_cpu_split_options(self) -> None:
        args = argparse.Namespace(
            server=Path("/tmp/llama-server"),
            model=Path("/tmp/model.gguf"),
            port=12355,
            extra_server_arg=[],
            cache_type_k="q8_0",
            cache_type_v="q4_0",
            batch_size=256,
            ubatch_size=256,
            parallel=1,
            kv_unified="auto",
            n_gpu_layers="all",
            kv_stream_cpu_threads=None,
            kv_stream_cpu_share=None,
        )
        command = BENCHMARK.server_command(args, 131072, 512)
        self.assertNotIn("--kv-stream-cpu-threads", command)
        self.assertNotIn("--kv-stream-cpu-share", command)
        args.kv_stream_cpu_threads = 6
        args.kv_stream_cpu_share = "auto"
        command = BENCHMARK.server_command(args, 131072, 512)
        self.assertEqual(command[command.index("--kv-stream-cpu-threads") + 1], "6")
        self.assertEqual(command[command.index("--kv-stream-cpu-share") + 1], "auto")

    def test_cpu_split_options_are_validated_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            server = root / "llama-server"
            model.touch()
            server.touch(mode=0o755)
            base = [
                "--model", str(model), "--server", str(server),
                "--max-context", "64K", "--min-context", "8K",
            ]
            for extra, message in (
                (["--kv-stream-cpu-threads", "-1"], "CPU threads"),
                (["--kv-stream-cpu-share", "1.5"], "CPU share"),
                (["--kv-stream-cpu-share", "fast"], "CPU share"),
                (["--server-env", "NOVALUE"], "server-env"),
            ):
                with self.assertRaisesRegex(SystemExit, message):
                    BENCHMARK.validate_args(BENCHMARK.parse_args(base + extra))
            args = BENCHMARK.parse_args(base + [
                "--kv-stream-cpu-threads", "6", "--kv-stream-cpu-share", "0.4",
                "--server-env", "GGML_CUDA_KV_STREAM_CPU_CPUS=2-7",
            ])
            BENCHMARK.validate_args(args)
            signature = BENCHMARK.resume_signature(args, [8192])
        self.assertEqual(signature["kv_stream_cpu_threads"], 6)
        self.assertEqual(signature["kv_stream_cpu_share"], "0.4")
        self.assertEqual(signature["server_env"], ["GGML_CUDA_KV_STREAM_CPU_CPUS=2-7"])

    def test_csv_lists_cpu_split_columns(self) -> None:
        rows = {
            8192: {
                "context_capacity": 8192,
                "stream_total_cpu_pages": 440,
                "stream_total_cpu_decline_below_min": 2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            BENCHMARK.write_csv(path, rows)
            header, row = path.read_text().splitlines()[:2]
        for field in (
            "stream_max_copy_busy",
            "stream_total_cpu_pages",
            "stream_total_cpu_decline_prefill",
            "stream_total_cpu_decline_no_eligible",
            "stream_total_cpu_decline_below_min",
        ):
            self.assertIn(field, header.split(","))
        self.assertIn("440", row.split(","))


if __name__ == "__main__":
    unittest.main()
