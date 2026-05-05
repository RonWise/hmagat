import argparse
import pickle

import numpy as np
import torch
from loguru import logger

from hmagat.convert_to_imitation_dataset import (
    add_imitation_dataset_args,
    get_imitation_dataset_file_name,
)
from hmagat.dataset_loading import load_dataset
from hmagat.downstream_shards import add_sharded_downstream_args
from hmagat.downstream_shards import load_stage_manifest, warn_and_raise
from hmagat.run_expert import add_expert_dataset_args


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _field_length(field):
    return int(field.shape[0]) if isinstance(field, torch.Tensor) else len(field)


def _field_item(field, index):
    return field[index]


def _values_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor):
            left = torch.as_tensor(left)
        if not isinstance(right, torch.Tensor):
            right = torch.as_tensor(right)
        return left.dtype == right.dtype and torch.equal(left, right)

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_values_equal(l_item, r_item) for l_item, r_item in zip(left, right))

    return left == right


def _assert_same_value(left, right, *, field_idx, global_idx):
    if not _values_equal(left, right):
        left_shape = getattr(left, "shape", None)
        right_shape = getattr(right, "shape", None)
        warn_and_raise(
            "Processed dataset mismatch: "
            f"field={field_idx}, snapshot={global_idx}, "
            f"left_type={type(left).__name__}, right_type={type(right).__name__}, "
            f"left_shape={left_shape}, right_shape={right_shape}."
        )


def _assert_same_schema(legacy_dataset, shard_dataset, shard_path):
    if len(legacy_dataset) != len(shard_dataset):
        warn_and_raise(
            "Processed dataset field count mismatch: "
            f"legacy={len(legacy_dataset)}, shard={len(shard_dataset)}, path={shard_path}."
        )


def compare_processed_dataset_shards(args):
    logger.warning(
        "Processed dataset comparison loads the legacy single-file processed dataset "
        "fully into RAM; use only when enough memory is available."
    )
    legacy_dataset = load_dataset(
        [get_imitation_dataset_file_name],
        "processed_dataset",
        args,
    )
    manifest = load_stage_manifest(args, "processed_dataset", expected_stage="processed")

    legacy_snapshots = _field_length(legacy_dataset[0])
    if legacy_snapshots != manifest["total_snapshots"]:
        warn_and_raise(
            "Processed dataset total snapshot mismatch: "
            f"legacy={legacy_snapshots}, shards={manifest['total_snapshots']}."
        )

    global_idx = 0
    for entry in manifest["entries"]:
        shard_dataset = _load_pickle(entry["path"])
        _assert_same_schema(legacy_dataset, shard_dataset, entry["path"])
        shard_snapshots = _field_length(shard_dataset[0])
        if shard_snapshots != entry["snapshot_count"]:
            warn_and_raise(
                "Processed shard snapshot count mismatch: "
                f"path={entry['path']}, data={shard_snapshots}, "
                f"manifest={entry['snapshot_count']}."
            )

        for local_idx in range(shard_snapshots):
            for field_idx, (legacy_field, shard_field) in enumerate(
                zip(legacy_dataset, shard_dataset)
            ):
                _assert_same_value(
                    _field_item(legacy_field, global_idx),
                    _field_item(shard_field, local_idx),
                    field_idx=field_idx,
                    global_idx=global_idx,
                )
            global_idx += 1

        logger.info(
            "Processed shard comparison passed: "
            f"shard={entry['shard_idx']}, snapshots={shard_snapshots}, "
            f"checked_total={global_idx}/{legacy_snapshots}"
        )

    if global_idx != legacy_snapshots:
        warn_and_raise(
            "Processed dataset checked snapshot mismatch: "
            f"checked={global_idx}, legacy={legacy_snapshots}."
        )

    logger.info(
        "Processed legacy dataset and processed shards are semantically identical: "
        f"snapshots={legacy_snapshots}, shards={len(manifest['entries'])}."
    )
    return {
        "snapshots": legacy_snapshots,
        "shards": len(manifest["entries"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare legacy processed dataset with processed dataset shards."
    )
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_sharded_downstream_args(parser)
    parser.add_argument("--use_edge_attr", action="store_true", default=False)
    args = parser.parse_args()

    compare_processed_dataset_shards(args)


if __name__ == "__main__":
    main()
