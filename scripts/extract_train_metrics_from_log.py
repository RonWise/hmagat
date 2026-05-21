#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


BATCH_PROGRESS_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| INFO\s+\| "
    r"hmagat\.progress_logging:update:\d+ - Training epoch (?P<epoch>\d+): "
    r"(?P<batch_idx>\d+)/(?P<total_batches>\d+) "
    r"\((?P<progress_pct>[\d.]+)%\), elapsed=(?P<elapsed>[^,]+), "
    r"eta=(?P<eta>[^,]+), loss=(?P<loss>[-+eE\d.]+)"
    r"(?:, accuracy=(?P<accuracy>[-+eE\d.]+))?$"
)

EPOCH_SUMMARY_RE = re.compile(
    r"^Epoch (?P<epoch>\d+), Mean Loss: (?P<mean_loss>[-+eE\d.]+) , "
    r"Mean Accuracy: (?P<mean_accuracy>[-+eE\d.]+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract batch-level train loss and epoch-level summaries from a training log."
    )
    parser.add_argument("log_path", type=Path)
    parser.add_argument(
        "--batch-loss-csv",
        type=Path,
        help="Output CSV path for batch-level train loss. Defaults next to the log.",
    )
    parser.add_argument(
        "--epoch-summary-csv",
        type=Path,
        help="Output CSV path for epoch-level mean loss/accuracy. Defaults next to the log.",
    )
    parser.add_argument(
        "--availability-csv",
        type=Path,
        help="Output CSV path describing which metrics are present in the log. Defaults next to the log.",
    )
    return parser.parse_args()


def default_output_paths(log_path: Path) -> tuple[Path, Path, Path]:
    stem = log_path.stem
    parent = log_path.parent
    return (
        parent / f"{stem}_batch_loss.csv",
        parent / f"{stem}_epoch_summary.csv",
        parent / f"{stem}_metric_availability.csv",
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    log_path = args.log_path
    batch_loss_csv, epoch_summary_csv, availability_csv = default_output_paths(log_path)
    if args.batch_loss_csv is not None:
        batch_loss_csv = args.batch_loss_csv
    if args.epoch_summary_csv is not None:
        epoch_summary_csv = args.epoch_summary_csv
    if args.availability_csv is not None:
        availability_csv = args.availability_csv

    batch_rows: list[dict] = []
    epoch_rows: list[dict] = []
    batch_accuracy_available = False

    with log_path.open(encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n")

            batch_match = BATCH_PROGRESS_RE.match(line)
            if batch_match:
                batch_rows.append(
                    {
                        "line_number": line_number,
                        "timestamp": batch_match.group("timestamp"),
                        "epoch": int(batch_match.group("epoch")),
                        "batch_idx": int(batch_match.group("batch_idx")),
                        "total_batches": int(batch_match.group("total_batches")),
                        "progress_pct": float(batch_match.group("progress_pct")),
                        "elapsed_text": batch_match.group("elapsed"),
                        "eta_text": batch_match.group("eta"),
                        "train_batch_loss": float(batch_match.group("loss")),
                        "train_batch_accuracy": (
                            ""
                            if batch_match.group("accuracy") is None
                            else float(batch_match.group("accuracy"))
                        ),
                    }
                )
                if batch_match.group("accuracy") is not None:
                    batch_accuracy_available = True
                continue

            epoch_match = EPOCH_SUMMARY_RE.match(line)
            if epoch_match:
                epoch_rows.append(
                    {
                        "line_number": line_number,
                        "epoch": int(epoch_match.group("epoch")),
                        "mean_train_loss": float(epoch_match.group("mean_loss")),
                        "mean_train_accuracy": float(epoch_match.group("mean_accuracy")),
                    }
                )

    write_csv(
        batch_loss_csv,
        [
            "line_number",
            "timestamp",
            "epoch",
            "batch_idx",
            "total_batches",
            "progress_pct",
            "elapsed_text",
            "eta_text",
            "train_batch_loss",
            "train_batch_accuracy",
        ],
        batch_rows,
    )
    write_csv(
        epoch_summary_csv,
        ["line_number", "epoch", "mean_train_loss", "mean_train_accuracy"],
        epoch_rows,
    )

    availability_rows = [
        {
            "metric_name": "train_batch_loss",
            "granularity": "batch",
            "source": "log",
            "available": "yes" if batch_rows else "no",
            "notes": "Extracted from progress lines with loss=...",
        },
        {
            "metric_name": "train_batch_success_or_accuracy",
            "granularity": "batch",
            "source": "log",
            "available": "yes" if batch_accuracy_available else "no",
            "notes": (
                "Extracted from progress lines with accuracy=..."
                if batch_accuracy_available
                else "This log format does not write batch-level success/accuracy."
            ),
        },
        {
            "metric_name": "mean_train_loss",
            "granularity": "epoch",
            "source": "log",
            "available": "yes" if epoch_rows else "no",
            "notes": "Would be extracted from `Epoch X, Mean Loss: ...` summary lines.",
        },
        {
            "metric_name": "mean_train_accuracy",
            "granularity": "epoch",
            "source": "log",
            "available": "yes" if epoch_rows else "no",
            "notes": "Would be extracted from `Mean Accuracy: ...` summary lines.",
        },
    ]
    write_csv(
        availability_csv,
        ["metric_name", "granularity", "source", "available", "notes"],
        availability_rows,
    )

    print(f"batch_loss_csv={batch_loss_csv}")
    print(f"epoch_summary_csv={epoch_summary_csv}")
    print(f"availability_csv={availability_csv}")
    print(f"batch_rows={len(batch_rows)}")
    print(f"epoch_rows={len(epoch_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
