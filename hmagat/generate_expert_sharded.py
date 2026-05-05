import argparse
import os
import pathlib
import subprocess
import sys
import time

from loguru import logger

from hmagat.expert_shards import merge_expert_shards, split_sample_ranges
from hmagat.run_expert import add_expert_dataset_args


RUNNER_ONLY_ARGS = {
    "logs_dir",
    "num_workers",
    "poll_seconds",
    "resource_monitor",
    "resource_monitor_interval",
}

SHARD_ARGS = {
    "sample_start",
    "sample_end",
    "shard_output_name",
}

BOOLEAN_OPTIONAL_ARGS = {
    "block_extra_space",
    "ensure_grid_config_is_generatable",
    "pibt_expert_relevance_training",
    "regulate_obstacle_density_max",
    "save_termination_state",
}


def _warn_and_raise(message, exc_type=ValueError):
    logger.warning(message)
    raise exc_type(message)


def _append_arg(command, key, value):
    arg_name = f"--{key}"
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(arg_name)
        elif key in BOOLEAN_OPTIONAL_ARGS:
            command.append(f"--no-{key}")
        return
    command.extend([arg_name, str(value)])


def _build_worker_command(args, shard_idx, sample_start, sample_end):
    command = [sys.executable, "-m", "hmagat.run_expert"]
    for key, value in vars(args).items():
        if key in RUNNER_ONLY_ARGS or key in SHARD_ARGS:
            continue
        _append_arg(command, key, value)

    shard_output_name = (
        f"{args.override_name or 'expert'}_raw_shard_{shard_idx:03d}_"
        f"{sample_start:05d}_{sample_end:05d}.pkl"
    )
    command.extend(
        [
            "--sample_start",
            str(sample_start),
            "--sample_end",
            str(sample_end),
            "--shard_output_name",
            shard_output_name,
        ]
    )
    return command


def _normalize_python_command(command):
    parts = command.strip().split(None, 1)
    if not parts:
        return command
    executable = pathlib.Path(parts[0]).name
    if executable.startswith("python"):
        suffix = f" {parts[1]}" if len(parts) > 1 else ""
        return f"python{suffix}"
    return command


def _is_tracked_resource_process(command, dataset_dir):
    if dataset_dir not in command:
        return False
    normalized = _normalize_python_command(command)
    return normalized.startswith(
        "python -m hmagat.run_expert "
    ) or normalized.startswith("python -m hmagat.generate_expert_sharded ")


def _read_mem_info_gib():
    values = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    total = values["MemTotal"]
    available = values.get("MemAvailable", values.get("MemFree", 0))
    used = total - available
    return total / 1024 / 1024, used / 1024 / 1024, 100.0 * used / total


def _read_tracked_processes(dataset_dir):
    output = subprocess.check_output(
        ["ps", "-eo", "pid,pcpu,pmem,rss,args", "--no-headers"],
        text=True,
    )
    processes = []
    for line in output.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, pcpu, _pmem, rss, command = parts
        if not _is_tracked_resource_process(command, dataset_dir):
            continue
        normalized = _normalize_python_command(command)
        processes.append(
            {
                "pid": int(pid),
                "pcpu": float(pcpu),
                "rss_mib": int(rss) / 1024,
                "is_worker": normalized.startswith("python -m hmagat.run_expert "),
            }
        )
    return processes


def _collect_resource_snapshot(dataset_dir):
    try:
        processes = _read_tracked_processes(str(dataset_dir))
        load1, load5, load15 = os.getloadavg()
        mem_total_gib, mem_used_gib, mem_used_percent = _read_mem_info_gib()
    except Exception as exc:
        logger.warning(f"Resource monitoring failed: {exc}")
        return None

    return {
        "workers": sum(1 for process in processes if process["is_worker"]),
        "tracked_processes": len(processes),
        "tracked_pcpu_sum": sum(process["pcpu"] for process in processes),
        "tracked_rss_mib": sum(process["rss_mib"] for process in processes),
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "mem_total_gib": mem_total_gib,
        "mem_used_gib": mem_used_gib,
        "mem_used_percent": mem_used_percent,
    }


def _log_resource_snapshot(dataset_dir):
    snapshot = _collect_resource_snapshot(dataset_dir)
    if snapshot is None:
        return
    logger.info(
        "Sharded expert resource usage: "
        f"workers={snapshot['workers']}, "
        f"tracked_processes={snapshot['tracked_processes']}, "
        f"tracked_cpu={snapshot['tracked_pcpu_sum']:.1f}%, "
        f"tracked_rss={snapshot['tracked_rss_mib']:.1f}MiB, "
        f"load1={snapshot['load1']:.2f}, "
        f"load5={snapshot['load5']:.2f}, "
        f"load15={snapshot['load15']:.2f}, "
        f"mem={snapshot['mem_used_gib']:.2f}/"
        f"{snapshot['mem_total_gib']:.2f}GiB "
        f"({snapshot['mem_used_percent']:.1f}%)"
    )


def _terminate_running_workers(workers):
    for worker in workers:
        process = worker["process"]
        if process.poll() is None:
            logger.warning(
                "Terminating expert shard worker after orchestration failure: "
                f"shard={worker['shard_idx']}, pid={process.pid}"
            )
            process.terminate()


def run_sharded_expert_generation(args):
    if args.resource_monitor_interval <= 0:
        _warn_and_raise(
            "Invalid resource monitor interval: "
            "resource_monitor_interval must be positive."
        )

    ranges = split_sample_ranges(args.num_samples, args.num_workers)

    logs_dir = pathlib.Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    workers = []
    for shard_idx, (sample_start, sample_end) in enumerate(ranges):
        log_path = logs_dir / f"expert_shard_{shard_idx:03d}.log"
        command = _build_worker_command(args, shard_idx, sample_start, sample_end)
        log_file = open(log_path, "w")
        process = subprocess.Popen(
            command,
            cwd=pathlib.Path(__file__).resolve().parents[1],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        workers.append(
            {
                "shard_idx": shard_idx,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "command": command,
                "log_path": log_path,
                "log_file": log_file,
                "process": process,
                "reported_done": False,
            }
        )
        logger.info(
            "Started expert shard worker: "
            f"shard={shard_idx}, range=[{sample_start}, {sample_end}), "
            f"pid={process.pid}, log={log_path}"
        )

    start_time = time.monotonic()
    last_resource_log_time = None
    try:
        while True:
            now = time.monotonic()
            completed = 0
            failed = []
            running = 0
            for worker in workers:
                process = worker["process"]
                returncode = process.poll()
                if returncode is None:
                    running += 1
                    continue
                completed += 1
                if returncode != 0:
                    failed.append(worker)
                elif not worker["reported_done"]:
                    worker["reported_done"] = True
                    logger.info(
                        "Expert shard worker completed: "
                        f"shard={worker['shard_idx']}, "
                        f"range=[{worker['sample_start']}, {worker['sample_end']}), "
                        f"log={worker['log_path']}"
                    )

            elapsed = int(time.monotonic() - start_time)
            logger.info(
                "Sharded expert generation progress: "
                f"running={running}, completed={completed}/{len(workers)}, "
                f"failed={len(failed)}, elapsed={elapsed}s"
            )
            if args.resource_monitor and (
                last_resource_log_time is None
                or now - last_resource_log_time >= args.resource_monitor_interval
            ):
                _log_resource_snapshot(args.dataset_dir)
                last_resource_log_time = now

            if failed:
                for worker in failed:
                    logger.warning(
                        "Expert shard worker failed: "
                        f"shard={worker['shard_idx']}, "
                        f"returncode={worker['process'].returncode}, "
                        f"log={worker['log_path']}"
                    )
                _terminate_running_workers(workers)
                raise RuntimeError("At least one expert shard worker failed.")

            if completed == len(workers):
                break

            time.sleep(args.poll_seconds)
    finally:
        for worker in workers:
            worker["log_file"].close()

    return merge_expert_shards(args, expected_num_shards=args.num_workers)


def main():
    parser = argparse.ArgumentParser(description="Generate expert dataset shards")
    parser = add_expert_dataset_args(parser)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--poll_seconds", type=float, default=30.0)
    parser.add_argument("--logs_dir", type=str, default=None)
    parser.add_argument(
        "--resource_monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resource_monitor_interval", type=float, default=30.0)

    args = parser.parse_args()
    if args.logs_dir is None:
        args.logs_dir = str(pathlib.Path(args.dataset_dir, "logs"))

    logger.info(args)
    run_sharded_expert_generation(args)


if __name__ == "__main__":
    main()
