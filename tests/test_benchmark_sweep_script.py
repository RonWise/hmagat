import json
import os
import pathlib
import shlex
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "run_hmagat_cs_4090_benchmark_sweep.sh"


class BenchmarkSweepScriptTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp.name)
        self.fakebin = self.tmp_path / "fakebin"
        self.fakebin.mkdir()
        self.record_dir = self.tmp_path / "records"
        self.record_dir.mkdir()
        self.workspace_dir = self.tmp_path / "workspace with spaces"
        self.workspace_dir.mkdir()
        self.log_root = self.tmp_path / "logs root"
        self.checkpoint_root = self.tmp_path / "checkpoints root"
        self.tensorboard_root = self.tmp_path / "runs root"

        self._write_fake_python()
        self._write_fake_docker()
        self._write_fake_nvidia_smi()
        self._write_fake_ps()

    def tearDown(self):
        self._tmp.cleanup()

    def _write_executable(self, path, content):
        path.write_text(content)
        path.chmod(0o755)

    def _write_fake_python(self):
        self._write_executable(
            self.fakebin / "python",
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                import pathlib
                import shlex
                import sys

                record_dir = pathlib.Path(os.environ["FAKE_DOCKER_RECORD_DIR"])
                with (record_dir / "python_calls.jsonl").open("a", encoding="utf-8") as f:
                    json.dump(sys.argv[1:], f)
                    f.write("\\n")

                if len(sys.argv) > 1 and sys.argv[1] == "-" and "BENCHMARK_CMD_SHELL" in os.environ:
                    cmd = shlex.split(os.environ["BENCHMARK_CMD_SHELL"])
                    with (record_dir / "training_calls.jsonl").open("a", encoding="utf-8") as f:
                        json.dump(cmd, f)
                        f.write("\\n")

                    monitor_path = pathlib.Path(os.environ["BENCHMARK_MONITOR_FILE"])
                    summary_path = pathlib.Path(os.environ["BENCHMARK_SUMMARY_FILE"])
                    log_path = pathlib.Path(os.environ["BENCHMARK_LOG_FILE"])

                    monitor_path.parent.mkdir(parents=True, exist_ok=True)
                    monitor_path.write_text(
                        "sample_idx,wall_time_iso,elapsed_sec,gpu_index,gpu_util_pct,gpu_mem_used_mb,gpu_mem_total_mb,proc_rss_kb,proc_cpu_pct,mem_total_kb,mem_available_kb\\n"
                        "1,2026-05-17T00:00:00,0.0,0,81,12000,24564,512000,190.5,1000000,400000\\n"
                        "2,2026-05-17T00:00:05,5.0,0,93,12800,24564,640000,210.0,1000000,380000\\n",
                        encoding="utf-8",
                    )

                    summary = {
                        "run_name": os.environ["BENCHMARK_RUN_NAME"],
                        "mode": os.environ["BENCHMARK_MODE"],
                        "batch_size": int(os.environ["BENCHMARK_BATCH_SIZE"]),
                        "truncated_bptt_length": int(os.environ["BENCHMARK_TBPTT"]),
                        "dataloader_num_workers": int(os.environ["BENCHMARK_WORKERS"]),
                        "dataloader_prefetch_factor": int(os.environ["BENCHMARK_PREFETCH"]),
                        "max_train_batches": int(os.environ["BENCHMARK_MAX_TRAIN_BATCHES"]),
                        "monitor_interval_secs": float(os.environ["BENCHMARK_MONITOR_INTERVAL_SECS"]),
                        "num_samples_logged": 2,
                        "exit_code": int(os.environ.get("FAKE_PYTHON_EXIT_CODE", "0")),
                        "elapsed_sec": 5.0,
                        "gpu_util_avg_pct": 87.0,
                        "gpu_util_peak_pct": 93.0,
                        "gpu_mem_used_avg_mb": 12400.0,
                        "gpu_mem_used_peak_mb": 12800.0,
                        "proc_rss_peak_kb": 640000,
                        "proc_cpu_avg_pct": 200.25,
                        "system_mem_used_peak_kb": 620000,
                    }
                    summary_path.write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\\n",
                        encoding="utf-8",
                    )
                    log_path.write_text("FAKE benchmark log\\n", encoding="utf-8")
                    print(json.dumps(summary, sort_keys=True))
                    sys.exit(int(os.environ.get("FAKE_PYTHON_EXIT_CODE", "0")))

                if len(sys.argv) > 1 and sys.argv[1] == "-":
                    with (record_dir / "preflight_calls.jsonl").open("a", encoding="utf-8") as f:
                        json.dump(sys.argv[1:], f)
                        f.write("\\n")
                    print(
                        os.environ.get(
                            "FAKE_PREFLIGHT_STDOUT",
                            "FAKE_PREFLIGHT_OK",
                        )
                    )
                    sys.exit(int(os.environ.get("FAKE_PREFLIGHT_EXIT_CODE", "0")))

                print("FAKE_PYTHON_OK")
                sys.exit(int(os.environ.get("FAKE_PYTHON_EXIT_CODE", "0")))
                """
            ),
        )

    def _write_fake_nvidia_smi(self):
        self._write_executable(
            self.fakebin / "nvidia-smi",
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                echo "0, 88, 12345, 24564"
                """
            ),
        )

    def _write_fake_ps(self):
        self._write_executable(
            self.fakebin / "ps",
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                if [[ "${1:-}" == "-o" ]]; then
                  echo "640000 200.0"
                  exit 0
                fi
                /bin/ps "$@"
                """
            ),
        )

    def _write_fake_docker(self):
        self._write_executable(
            self.fakebin / "docker",
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -euo pipefail
                record_dir="${FAKE_DOCKER_RECORD_DIR:?}"
                fakebin="${FAKE_DOCKER_FAKEBIN:?}"
                cmd="${@: -1}"
                printf '%s\\0' "$cmd" >> "${record_dir}/docker_commands.bin"
                PATH="${fakebin}:$PATH" \
                FAKE_DOCKER_RECORD_DIR="${record_dir}" \
                /bin/bash -lc "$cmd"
                """
            ),
        )

    def _run_script(self, **overrides):
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{self.fakebin}{os.pathsep}{env['PATH']}",
                "FAKE_DOCKER_RECORD_DIR": str(self.record_dir),
                "FAKE_DOCKER_FAKEBIN": str(self.fakebin),
                "CONTAINER_NAME": "fake-hmagat-work",
                "WORKSPACE_DIR": str(self.workspace_dir),
                "LOG_ROOT": str(self.log_root),
                "CHECKPOINT_ROOT": str(self.checkpoint_root),
                "TENSORBOARD_ROOT": str(self.tensorboard_root),
                "SWEEP_PROFILE": "quick",
            }
        )
        env.update({key: str(value) for key, value in overrides.items()})
        return subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

    def _read_python_calls(self):
        path = self.record_dir / "python_calls.jsonl"
        if not path.exists():
            return []
        calls = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                calls.append(json.loads(line))
        return calls

    def _read_training_calls(self):
        path = self.record_dir / "training_calls.jsonl"
        if not path.exists():
            return []
        calls = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                calls.append(json.loads(line))
        return calls

    def _read_preflight_calls(self):
        path = self.record_dir / "preflight_calls.jsonl"
        if not path.exists():
            return []
        calls = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                calls.append(json.loads(line))
        return calls

    def _read_docker_commands(self):
        path = self.record_dir / "docker_commands.bin"
        if not path.exists():
            return []
        raw = path.read_bytes()
        return [chunk.decode("utf-8") for chunk in raw.split(b"\0") if chunk]

    def _read_summary(self, run_name):
        path = self.log_root / f"{run_name}.summary.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _flag_value(self, argv, flag):
        idx = argv.index(flag)
        return argv[idx + 1]

    def test_realistic_short_quick_sweep_runs_six_cases(self):
        result = self._run_script()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        calls = self._read_training_calls()
        self.assertEqual(len(calls), 6)
        first_call = calls[0]
        self.assertEqual(len(self._read_preflight_calls()), 1)

        self.assertEqual(first_call[:3], ["python", "-m", "hmagat.train_imitation_learning_pyg"])
        self.assertEqual(self._flag_value(first_call, "--collision_shielding"), "pibt")
        self.assertEqual(self._flag_value(first_call, "--action_sampling"), "probabilistic")
        self.assertEqual(self._flag_value(first_call, "--validation_every_epochs"), "1")
        self.assertEqual(self._flag_value(first_call, "--max_train_batches"), "80")
        self.assertEqual(self._flag_value(first_call, "--initial_val_size"), "16")
        self.assertIn("--tensorboard_dir", first_call)

        docker_commands = self._read_docker_commands()
        self.assertEqual(len(docker_commands), 8)
        self.assertIn("Running preflight checks", result.stdout)
        self.assertNotIn("/usr/bin/time", docker_commands[2])
        self.assertIn("BENCHMARK_MONITOR_FILE", docker_commands[2])
        self.assertIn("BENCHMARK_CMD_SHELL", docker_commands[2])

        summary = self._read_summary("bench_4090_L2")
        self.assertEqual(summary["gpu_util_avg_pct"], 87.0)
        self.assertEqual(summary["gpu_mem_used_peak_mb"], 12800.0)
        self.assertEqual(summary["proc_rss_peak_kb"], 640000)
        self.assertEqual(summary["mode"], "realistic_short")
        monitor_csv = self.log_root / "bench_4090_L2.monitor.csv"
        self.assertTrue(monitor_csv.exists())
        self.assertIn("gpu_util_pct", monitor_csv.read_text(encoding="utf-8"))

        aggregate_csv = self.log_root / "benchmark_sweep_summary.csv"
        aggregate_ranking = self.log_root / "benchmark_sweep_ranking.txt"
        self.assertTrue(aggregate_csv.exists())
        self.assertTrue(aggregate_ranking.exists())
        self.assertIn("Top successful candidate:", result.stdout)

    def test_inner_failure_propagates_and_stops_after_first_case(self):
        result = self._run_script(FAKE_PYTHON_EXIT_CODE=17)

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self._read_preflight_calls()), 1)
        self.assertEqual(len(self._read_training_calls()), 1)
        self.assertEqual(len(self._read_docker_commands()), 3)
        self.assertNotIn("Benchmark sweep finished successfully.", result.stdout)

    def test_dataset_overrides_with_spaces_survive_as_single_arguments(self):
        dataset_dir = self.tmp_path / "dataset dir with spaces"
        dataset_name = "hmagat cs spaced name"

        result = self._run_script(
            DATASET_DIR=dataset_dir,
            DATASET_NAME=dataset_name,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        first_call = self._read_training_calls()[0]
        self.assertEqual(self._flag_value(first_call, "--dataset_dir"), str(dataset_dir))
        self.assertEqual(self._flag_value(first_call, "--override_name"), dataset_name)

    def test_throughput_mode_disables_validation_and_intermediate_artifacts(self):
        result = self._run_script(BENCHMARK_MODE="throughput")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        first_call = self._read_training_calls()[0]
        self.assertIn("--skip_validation", first_call)
        self.assertIn("--no-save_intmd_checkpoints", first_call)
        self.assertNotIn("--tensorboard_dir", first_call)
        summary = self._read_summary("bench_4090_L2")
        self.assertEqual(summary["mode"], "throughput")

    def test_realistic_mode_has_distinct_long_run_defaults(self):
        result = self._run_script(BENCHMARK_MODE="realistic")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        first_call = self._read_training_calls()[0]
        self.assertEqual(self._flag_value(first_call, "--num_epochs"), "4")
        self.assertEqual(self._flag_value(first_call, "--validation_every_epochs"), "4")
        self.assertEqual(self._flag_value(first_call, "--max_train_batches"), "200")
        self.assertEqual(self._flag_value(first_call, "--initial_val_size"), "128")

    def test_stale_training_index_preflight_stops_before_first_benchmark_case(self):
        result = self._run_script(
            FAKE_PREFLIGHT_EXIT_CODE=2,
            FAKE_PREFLIGHT_STDOUT="STALE_INDEX_DETECTED",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self._read_preflight_calls()), 1)
        self.assertEqual(len(self._read_training_calls()), 0)
        self.assertEqual(len(self._read_docker_commands()), 1)
        self.assertIn("Running preflight checks...", result.stdout)
        self.assertIn("STALE_INDEX_DETECTED", result.stdout)


if __name__ == "__main__":
    unittest.main()
