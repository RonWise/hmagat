import argparse
import pathlib
import re
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a demo seed sweep and select the best seed."
    )
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--seed-step", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--results-tsv", type=str, required=True)
    parser.add_argument(
        "--sweep-arg",
        type=str,
        default="test_dataset_seed",
        help="Name of the CLI argument whose value should be swept over seeds.",
    )
    parser.add_argument(
        "--demo-script",
        type=str,
        default="test_imitation_learning_pyg.py",
        help="Path to the demo evaluation script relative to cwd.",
    )
    return parser.parse_known_args()


def grab_metric(text: str, pattern: str, name: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise RuntimeError(f"Could not parse {name} from demo output.")
    return float(match.group(1))


def main():
    args, demo_args = parse_args()
    rows = []

    for idx in range(args.num_seeds):
        seed = args.start_seed + idx * args.seed_step
        cmd = [
            sys.executable,
            args.demo_script,
            *demo_args,
            f"--{args.sweep_arg}",
            str(seed),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise RuntimeError(
                f"Demo run failed for seed {seed} with exit code {result.returncode}.\n"
                f"{output}"
            )

        success_rate = grab_metric(output, r"Success Rate:\s*([0-9.]+)", "success rate")
        makespan = grab_metric(output, r"Average Makespan:\s*([0-9.]+)", "makespan")
        partial_success = grab_metric(
            output,
            r"Average Partial Success Rate:\s*([0-9.]+)",
            "partial success rate",
        )
        sum_of_costs = grab_metric(
            output, r"Average Sum of Costs:\s*([0-9.]+)", "sum of costs"
        )
        row = {
            "seed": seed,
            "success_rate": success_rate,
            "partial_success_rate": partial_success,
            "sum_of_costs": sum_of_costs,
            "makespan": makespan,
        }
        rows.append(row)
        print(
            f"{seed}\t{success_rate}\t{partial_success}\t{sum_of_costs}\t{makespan}",
            flush=True,
        )

    rows.sort(
        key=lambda row: (
            -row["success_rate"],
            -row["partial_success_rate"],
            row["sum_of_costs"],
            row["makespan"],
            row["seed"],
        )
    )

    results_path = pathlib.Path(args.results_tsv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "seed\tsuccess_rate\tpartial_success_rate\tsum_of_costs\tmakespan",
        *[
            (
                f"{row['seed']}\t{row['success_rate']}\t"
                f"{row['partial_success_rate']}\t{row['sum_of_costs']}\t"
                f"{row['makespan']}"
            )
            for row in rows
        ],
    ]
    results_path.write_text("\n".join(lines) + "\n")

    best = rows[0]
    print("---BEST---", flush=True)
    print(
        (
            f"{best['seed']}\t{best['success_rate']}\t"
            f"{best['partial_success_rate']}\t{best['sum_of_costs']}\t"
            f"{best['makespan']}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
