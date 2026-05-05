import pathlib
import pickle

import numpy as np
from loguru import logger

from hmagat.run_expert import get_expert_dataset_file_name


SHARD_SPECIFIC_ARG_KEYS = {
    "sample_start",
    "sample_end",
    "shard_output_name",
}


def _warn_and_raise(message, exc_type=ValueError):
    logger.warning(message)
    raise exc_type(message)


def build_global_seeds(dataset_seed, num_samples):
    return np.random.default_rng(dataset_seed).integers(10**10, size=num_samples)


def split_sample_ranges(num_samples, num_shards):
    if num_samples <= 0:
        _warn_and_raise("Cannot split expert samples: num_samples must be positive.")
    if num_shards <= 0:
        _warn_and_raise("Cannot split expert samples: num_shards must be positive.")
    if num_shards > num_samples:
        _warn_and_raise(
            "Cannot split expert samples: num_shards must not exceed num_samples "
            f"({num_shards} > {num_samples})."
        )

    base = num_samples // num_shards
    remainder = num_samples % num_shards
    ranges = []
    start = 0
    for shard_idx in range(num_shards):
        size = base + (1 if shard_idx < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def validate_sample_range(sample_start, sample_end, num_samples):
    if sample_start is None and sample_end is None:
        return 0, num_samples
    if sample_start is None or sample_end is None:
        _warn_and_raise(
            "Expert shard range is incomplete: both sample_start and sample_end "
            "must be provided."
        )
    if not (0 <= sample_start < sample_end <= num_samples):
        _warn_and_raise(
            "Invalid expert shard range: expected "
            f"0 <= sample_start < sample_end <= num_samples, got "
            f"sample_start={sample_start}, sample_end={sample_end}, "
            f"num_samples={num_samples}."
        )
    return sample_start, sample_end


def default_shard_output_name(args, sample_start, sample_end):
    base_name = get_expert_dataset_file_name(args)
    stem = pathlib.Path(base_name).stem
    width = max(5, len(str(args.num_samples)))
    return f"{stem}_raw_shard_{sample_start:0{width}d}_{sample_end:0{width}d}.pkl"


def is_shard_mode(args):
    return args.sample_start is not None or args.sample_end is not None


def get_shards_dir(args):
    return pathlib.Path(args.dataset_dir, "raw_expert_predictions", "shards")


def get_shard_path(args, shard_output_name):
    return get_shards_dir(args) / shard_output_name


def _metadata_args(args):
    return {
        key: value
        for key, value in vars(args).items()
        if key not in SHARD_SPECIFIC_ARG_KEYS
    }


def write_expert_shard(
    args,
    *,
    dataset,
    seed_mask,
    sample_start,
    sample_end,
    shard_output_name=None,
):
    if len(seed_mask) != sample_end - sample_start:
        _warn_and_raise(
            "Cannot write expert shard: seed_mask length does not match shard "
            f"range length ({len(seed_mask)} vs {sample_end - sample_start})."
        )

    if shard_output_name is None:
        shard_output_name = default_shard_output_name(args, sample_start, sample_end)

    path = get_shard_path(args, shard_output_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "seed_mask": list(seed_mask),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "num_samples": args.num_samples,
        "dataset_seed": args.dataset_seed,
        "args": _metadata_args(args),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(
        "Expert shard written: "
        f"path={path}, range=[{sample_start}, {sample_end}), "
        f"saved_samples={len(dataset)}, requested_samples={len(seed_mask)}"
    )
    return str(path)


def _load_shard(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    required_keys = {
        "dataset",
        "seed_mask",
        "sample_start",
        "sample_end",
        "num_samples",
        "dataset_seed",
        "args",
    }
    missing = required_keys - set(payload)
    if missing:
        _warn_and_raise(
            f"Invalid expert shard {path}: missing required keys {sorted(missing)}."
        )
    return payload


def _validate_shard_payload(payload, args, path):
    if payload["num_samples"] != args.num_samples:
        _warn_and_raise(
            f"Expert shard metadata mismatch in {path}: num_samples "
            f"{payload['num_samples']} != {args.num_samples}."
        )
    if payload["dataset_seed"] != args.dataset_seed:
        _warn_and_raise(
            f"Expert shard metadata mismatch in {path}: dataset_seed "
            f"{payload['dataset_seed']} != {args.dataset_seed}."
        )

    seed_mask = payload["seed_mask"]
    sample_start = payload["sample_start"]
    sample_end = payload["sample_end"]
    if len(seed_mask) != sample_end - sample_start:
        _warn_and_raise(
            f"Invalid expert shard {path}: seed_mask length {len(seed_mask)} "
            f"does not match range [{sample_start}, {sample_end})."
        )

    if len(payload["dataset"]) != sum(bool(value) for value in seed_mask):
        _warn_and_raise(
            f"Invalid expert shard {path}: dataset length {len(payload['dataset'])} "
            "does not match the number of successful seed_mask entries."
        )


def _validate_compatible_generation_args(shards):
    reference_args = shards[0]["payload"].get("args", {})
    for shard in shards[1:]:
        shard_args = shard["payload"].get("args", {})
        if shard_args != reference_args:
            _warn_and_raise(
                "Expert shard generation args mismatch: "
                f"{shard['path']} differs from {shards[0]['path']}."
            )


def _list_shard_paths(args):
    shards_dir = get_shards_dir(args)
    return sorted(shards_dir.glob("*.pkl"))


def merge_expert_shards(args, expected_num_shards=None):
    shard_paths = _list_shard_paths(args)
    if expected_num_shards is not None and len(shard_paths) != expected_num_shards:
        _warn_and_raise(
            "Expert shard count mismatch: expected "
            f"{expected_num_shards}, found {len(shard_paths)} in {get_shards_dir(args)}."
        )
    if not shard_paths:
        _warn_and_raise(f"No expert shards found in {get_shards_dir(args)}.")

    shards = []
    for path in shard_paths:
        payload = _load_shard(path)
        _validate_shard_payload(payload, args, path)
        shards.append({"path": str(path), "payload": payload})

    shards.sort(key=lambda shard: shard["payload"]["sample_start"])
    _validate_compatible_generation_args(shards)

    expected_start = 0
    merged_dataset = []
    merged_seed_mask = []
    for shard in shards:
        payload = shard["payload"]
        sample_start = payload["sample_start"]
        sample_end = payload["sample_end"]
        if sample_start < expected_start:
            _warn_and_raise(
                "Expert shard overlap detected: "
                f"{shard['path']} starts at {sample_start}, expected "
                f"{expected_start}."
            )
        if sample_start > expected_start:
            _warn_and_raise(
                "Expert shard gap detected: "
                f"{shard['path']} starts at {sample_start}, expected "
                f"{expected_start}."
            )

        merged_dataset.extend(payload["dataset"])
        merged_seed_mask.extend(bool(value) for value in payload["seed_mask"])
        expected_start = sample_end

    if expected_start != args.num_samples:
        _warn_and_raise(
            "Expert shard coverage is incomplete: covered "
            f"{expected_start}/{args.num_samples} samples."
        )

    output_path = pathlib.Path(
        args.dataset_dir,
        "raw_expert_predictions",
        get_expert_dataset_file_name(args),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((merged_dataset, merged_seed_mask), f)

    logger.info(
        "Expert shards merged: "
        f"requested_samples={args.num_samples}, "
        f"covered_samples={len(merged_seed_mask)}, "
        f"successful_samples={len(merged_dataset)}, "
        f"failed_samples={args.num_samples - len(merged_dataset)}, "
        f"success_rate={len(merged_dataset) / args.num_samples:.6f}, "
        f"output_path={output_path}"
    )
    return str(output_path)
