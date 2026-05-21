import argparse
import csv
import pathlib
import subprocess
import sys


COMMON_DEMO_ARGS = [
    "--obs_radius",
    "5",
    "--save_termination_state",
    "--add_data_cost_to_go",
    "--normalize_cost_to_go",
    "--clamp_cost_to_go",
    "1.0",
    "--use_lists",
    "--run_online_expert",
    "--model_residuals",
    "all",
    "--use_edge_attr",
    "--use_edge_attr_for_messages",
    "positions+manhattan",
    "--edge_attr_cnn_mode",
    "MLP",
    "--load_positions_separately",
    "--train_on_terminated_agents",
    "--recursive_oe",
    "--cnn_mode",
    "ResNetLarge_withMLP",
    "--collision_shielding",
    "pibt",
    "--action_sampling",
    "probabilistic",
    "--test_name",
    "one_demo",
    "--test_num_samples",
    "1",
    "--test_obs_radius",
    "5",
    "--test_map_types",
    "warehouse=1.0",
    "--test_num_agents",
    "32+32",
    "--test_wall_width_min",
    "8",
    "--test_wall_width_max",
    "8",
    "--test_vertical_gap",
    "1",
    "--test_num_wall_rows_min",
    "5",
    "--test_num_wall_rows_max",
    "5",
    "--test_num_wall_cols_min",
    "2",
    "--test_num_wall_cols_max",
    "2",
    "--test_side_pad",
    "3",
    "--test_min_dist",
    "10",
]

HMAGAT_FAMILY_ARGS = [
    "--hypergraph_comm_radius",
    "7",
    "--hyperedge_generation_method",
    "kmeans",
    "--hypergraph_num_updates",
    "10",
    "--hypergraph_wait_one",
    "--hypergraph_initial_colperc",
    "0.1",
    "--hypergraph_final_colperc",
    "0.1",
    "--imitation_learning_model",
    "DirectionalHMAGAT",
    "--hyperedge_feature_generator",
    "magat",
    "--final_feature_generator",
    "magat",
    "--rl_based_temperature_sampling",
    "--temperature_checkpoints_dir",
    "checkpoints/hmagat_temperature_module",
    "--temperature_run_name",
    "simple_rl",
    "--temperature_actor_critic",
    "simple-local-val-init",
    "--temperature_optimize",
    "only-all-on-goal",
    "--iterations_per_epoch",
    "3",
    "--temperature_min_val",
    "0.5",
    "--temperature_max_val",
    "0.9",
    "--temperature_sampling_model_epoch_num",
    "43",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MAGAT, HMAGAT and HMAGAT-CS on identical demo seeds and collect "
            "step-by-step success-fraction curves into CSV."
        )
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary-csv", default=None)
    parser.add_argument("--fixed-instance-seed", type=int, default=7739560485)
    parser.add_argument("--sampling-seeds", type=str, default=None)
    parser.add_argument("--sampling-seed-base", type=int, default=42)
    parser.add_argument("--sampling-seed-step", type=int, default=10000)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--test-max-episode-steps", type=int, default=400)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--hmagat-cs-checkpoints-dir",
        default="checkpoints/hmagat_cs_sequence_32_scratch",
    )
    parser.add_argument("--hmagat-cs-model-epoch-num", type=int, default=15)
    return parser.parse_args()


def _parse_sampling_seeds(args):
    if args.sampling_seeds:
        return [int(part.strip()) for part in args.sampling_seeds.split(",") if part.strip()]
    return [
        int(args.sampling_seed_base + args.sampling_seed_step * idx)
        for idx in range(args.num_seeds)
    ]


def _python_executable():
    return sys.executable or "python"


def _run_eval(command):
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _model_commands(args, output_csv, sampling_seed):
    common_runtime = [
        "--device",
        str(args.device),
        "--test_max_episode_steps",
        str(args.test_max_episode_steps),
        "--test_fixed_instance_seed",
        str(args.fixed_instance_seed),
        "--test_sampling_seed",
        str(sampling_seed),
        "--step_metrics_csv_path",
        str(output_csv),
        "--step_metrics_pad_to_max_episode_steps",
        "--subsample_n",
        "1",
    ]

    magat_cmd = [
        _python_executable(),
        "test_imitation_learning_pyg.py",
        *COMMON_DEMO_ARGS,
        *common_runtime,
        "--step_metrics_model_label",
        "MAGAT",
        "--checkpoints_dir",
        "checkpoints/magat",
        "--run_name",
        "magat",
        "--imitation_learning_model",
        "MAGAT",
    ]

    hmagat_cmd = [
        _python_executable(),
        "test_imitation_learning_pyg.py",
        *COMMON_DEMO_ARGS,
        *common_runtime,
        "--step_metrics_model_label",
        "HMAGAT",
        "--checkpoints_dir",
        "checkpoints/hmagat",
        "--run_name",
        "hmagat",
        *HMAGAT_FAMILY_ARGS,
    ]

    hmagat_cs_cmd = [
        _python_executable(),
        "test_imitation_learning_pyg.py",
        *COMMON_DEMO_ARGS,
        *common_runtime,
        "--step_metrics_model_label",
        "HMAGAT-CS",
        "--checkpoints_dir",
        args.hmagat_cs_checkpoints_dir,
        "--run_name",
        "hmagat_cs",
        "--coordination_state_size",
        "32",
        "--model_epoch_num",
        str(args.hmagat_cs_model_epoch_num),
        *HMAGAT_FAMILY_ARGS,
    ]

    return [magat_cmd, hmagat_cmd, hmagat_cs_cmd]


def _write_summary_csv(output_csv, output_summary_csv):
    output_csv = pathlib.Path(output_csv)
    summary_path = pathlib.Path(output_summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = {}
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], int(row["sampling_seed"]))
            grouped.setdefault(key, []).append(row)

    fieldnames = [
        "model",
        "sampling_seed",
        "final_success_fraction",
        "rollout_success",
        "makespan",
        "first_step_reaching_0_5",
        "first_step_reaching_0_75",
        "first_step_reaching_0_9",
        "first_step_full_success",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, sampling_seed), rows in sorted(grouped.items()):
            rows = sorted(rows, key=lambda row: int(row["step"]))
            values = [float(row["success_fraction"]) for row in rows]

            def first_step(threshold):
                for row, value in zip(rows, values):
                    if value >= threshold:
                        return int(row["step"])
                return ""

            writer.writerow(
                {
                    "model": model,
                    "sampling_seed": sampling_seed,
                    "final_success_fraction": values[-1],
                    "rollout_success": int(rows[-1]["rollout_success"]),
                    "makespan": int(rows[-1]["makespan"]),
                    "first_step_reaching_0_5": first_step(0.5),
                    "first_step_reaching_0_75": first_step(0.75),
                    "first_step_reaching_0_9": first_step(0.9),
                    "first_step_full_success": first_step(1.0),
                }
            )


def main():
    args = parse_args()
    output_csv = pathlib.Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    sampling_seeds = _parse_sampling_seeds(args)
    print(
        f"Running demo success sweep on fixed_instance_seed={args.fixed_instance_seed} "
        f"for sampling_seeds={sampling_seeds}",
        flush=True,
    )

    for sampling_seed in sampling_seeds:
        for command in _model_commands(args, output_csv, sampling_seed):
            _run_eval(command)

    summary_path = args.output_summary_csv
    if summary_path is None:
        summary_path = str(output_csv.with_name(f"{output_csv.stem}_summary.csv"))
    _write_summary_csv(output_csv, summary_path)
    print(f"step_metrics_csv={output_csv}", flush=True)
    print(f"summary_csv={summary_path}", flush=True)


if __name__ == "__main__":
    main()
