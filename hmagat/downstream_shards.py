import json
import pathlib
import pickle

import numpy as np
from loguru import logger

from hmagat.expert_shards import build_global_seeds


COMMON_GENERATION_ARG_KEYS = (
    "obs_radius",
    "save_termination_state",
    "expert_algorithm",
    "obstacle_density_max",
    "ensure_grid_config_is_generatable",
)

STAGE_GENERATION_ARG_KEYS = {
    "processed": (
        "comm_radius",
        "num_neighbour_cutoff",
        "neighbour_cutoff_method",
        "use_lists",
        "distance_metric",
        "random_edge_probs",
        "use_edge_attr",
    ),
    "hypergraphs": (
        "hypergraph_comm_radius",
        "take_all_seeds",
        "hypergraph_max_group_size",
        "hyperedge_generation_method",
        "add_hypergraph_self_loop",
        "comm_self",
        "hypergraph_max_neighbours",
        "hypergraph_max_dist_threshold",
        "hypergraph_max_dist_frac",
        "hypergraph_max_clique_size",
        "hypergraph_initial_colperc",
        "hypergraph_final_colperc",
        "hypergraph_wait_one",
        "hypergraph_time_period",
        "hypergraph_num_updates",
    ),
    "additional_data": (
        "add_data_num_previous_actions",
        "add_data_cost_to_go",
        "normalize_cost_to_go",
        "clamp_cost_to_go",
        "add_data_greedy_action",
        "clamped_values_doubled",
    ),
    "positions": (
        "comm_radius",
        "num_neighbour_cutoff",
        "neighbour_cutoff_method",
        "use_lists",
        "use_edge_attr",
    ),
}


def warn_and_raise(message, exc_type=ValueError):
    logger.warning(message)
    raise exc_type(message)


def add_sharded_downstream_args(parser):
    parser.add_argument(
        "--use_shards",
        action="store_true",
        default=False,
        help="Read/write dataset stage shards instead of legacy single-file artifacts.",
    )
    return parser


def raw_expert_shards_dir(args):
    return pathlib.Path(args.dataset_dir, "raw_expert_predictions", "shards")


def stage_shards_dir(args, stage_dir_name):
    return pathlib.Path(args.dataset_dir, stage_dir_name, "shards")


def manifest_path(args, stage_dir_name):
    return stage_shards_dir(args, stage_dir_name) / "manifest.json"


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _write_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _jsonable(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def generation_args(args, stage):
    keys = COMMON_GENERATION_ARG_KEYS + STAGE_GENERATION_ARG_KEYS.get(stage, ())
    return {
        key: _jsonable(getattr(args, key))
        for key in keys
        if hasattr(args, key)
    }


def _validate_raw_payload(path, payload, args):
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
        warn_and_raise(
            f"Invalid raw expert shard {path}: missing required keys {sorted(missing)}."
        )
    if payload["num_samples"] != args.num_samples:
        warn_and_raise(
            f"Raw expert shard metadata mismatch in {path}: num_samples "
            f"{payload['num_samples']} != {args.num_samples}."
        )
    if payload["dataset_seed"] != args.dataset_seed:
        warn_and_raise(
            f"Raw expert shard metadata mismatch in {path}: dataset_seed "
            f"{payload['dataset_seed']} != {args.dataset_seed}."
        )
    sample_start = payload["sample_start"]
    sample_end = payload["sample_end"]
    seed_mask = payload["seed_mask"]
    if len(seed_mask) != sample_end - sample_start:
        warn_and_raise(
            f"Invalid raw expert shard {path}: seed_mask length {len(seed_mask)} "
            f"does not match range [{sample_start}, {sample_end})."
        )
    if len(payload["dataset"]) != sum(bool(value) for value in seed_mask):
        warn_and_raise(
            f"Invalid raw expert shard {path}: dataset length {len(payload['dataset'])} "
            "does not match successful seed_mask entries."
        )


def _raw_expert_shard_paths(args):
    paths = sorted(raw_expert_shards_dir(args).glob("*.pkl"))
    if not paths:
        warn_and_raise(f"No raw expert shards found in {raw_expert_shards_dir(args)}.")
    return paths


def iter_raw_expert_shards(args):
    expected_start = 0
    for path in _raw_expert_shard_paths(args):
        payload = _load_pickle(path)
        _validate_raw_payload(path, payload, args)
        sample_start = payload["sample_start"]
        sample_end = payload["sample_end"]
        if sample_start != expected_start:
            warn_and_raise(
                "Raw expert shard coverage is not contiguous: "
                f"{path} starts at {sample_start}, expected {expected_start}."
            )
        expected_start = sample_end
        yield {"path": str(path), "payload": payload}

    if expected_start != args.num_samples:
        warn_and_raise(
            "Raw expert shard coverage is incomplete: covered "
            f"{expected_start}/{args.num_samples} samples."
        )


def load_raw_expert_shards(args):
    return list(iter_raw_expert_shards(args))


def shard_output_name(args, stage, shard_idx, sample_start, sample_end):
    stem = args.override_name or "dataset"
    width = max(5, len(str(args.num_samples)))
    return (
        f"{stem}_{stage}_shard_{shard_idx:03d}_"
        f"{sample_start:0{width}d}_{sample_end:0{width}d}.pkl"
    )


def write_stage_shard(
    args,
    *,
    stage_dir_name,
    stage,
    shard_idx,
    sample_start,
    sample_end,
    payload,
    saved_samples,
    snapshot_count,
    graph_map_id_start=None,
    graph_map_id_end=None,
    original_sample_ids=None,
):
    output_dir = stage_shards_dir(args, stage_dir_name)
    path = output_dir / shard_output_name(args, stage, shard_idx, sample_start, sample_end)
    _write_pickle(path, payload)
    entry = {
        "stage": stage,
        "path": str(path),
        "file_name": path.name,
        "shard_idx": shard_idx,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "saved_samples": int(saved_samples),
        "snapshot_count": int(snapshot_count),
        "graph_map_id_start": graph_map_id_start,
        "graph_map_id_end": graph_map_id_end,
        "original_sample_ids": _jsonable(original_sample_ids),
        "num_samples": args.num_samples,
        "dataset_seed": args.dataset_seed,
        "override_name": args.override_name,
    }
    logger.info(
        "Dataset stage shard written: "
        f"stage={stage}, shard={shard_idx}, range=[{sample_start}, {sample_end}), "
        f"saved_samples={saved_samples}, snapshots={snapshot_count}, path={path}"
    )
    return entry


def write_stage_manifest(args, stage_dir_name, stage, entries):
    if not entries:
        warn_and_raise(f"Cannot write empty sharded manifest for stage {stage}.")
    output_path = manifest_path(args, stage_dir_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "stage": stage,
        "stage_dir_name": stage_dir_name,
        "override_name": args.override_name,
        "num_samples": args.num_samples,
        "dataset_seed": args.dataset_seed,
        "generation_args": generation_args(args, stage),
        "total_saved_samples": sum(entry["saved_samples"] for entry in entries),
        "total_snapshots": sum(entry["snapshot_count"] for entry in entries),
        "entries": entries,
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    logger.info(
        "Dataset stage manifest written: "
        f"stage={stage}, shards={len(entries)}, "
        f"total_snapshots={manifest['total_snapshots']}, path={output_path}"
    )
    return manifest


def load_stage_manifest(args, stage_dir_name, expected_stage=None):
    path = manifest_path(args, stage_dir_name)
    if not path.exists():
        warn_and_raise(f"Sharded manifest not found: {path}.")
    with open(path) as f:
        manifest = json.load(f)
    if expected_stage is not None and manifest.get("stage") != expected_stage:
        warn_and_raise(
            f"Sharded manifest stage mismatch in {path}: "
            f"{manifest.get('stage')} != {expected_stage}."
        )
    if manifest.get("num_samples") != args.num_samples:
        warn_and_raise(
            f"Sharded manifest num_samples mismatch in {path}: "
            f"{manifest.get('num_samples')} != {args.num_samples}."
        )
    if manifest.get("dataset_seed") != args.dataset_seed:
        warn_and_raise(
            f"Sharded manifest dataset_seed mismatch in {path}: "
            f"{manifest.get('dataset_seed')} != {args.dataset_seed}."
        )
    if manifest.get("override_name") != args.override_name:
        warn_and_raise(
            f"Sharded manifest override_name mismatch in {path}: "
            f"{manifest.get('override_name')} != {args.override_name}."
        )

    stored_generation_args = manifest.get("generation_args")
    if stored_generation_args is None:
        warn_and_raise(f"Sharded manifest lacks generation_args: {path}.")
    expected_generation_args = generation_args(args, manifest.get("stage"))
    missing_keys = sorted(set(expected_generation_args) - set(stored_generation_args))
    if missing_keys:
        warn_and_raise(
            f"Sharded manifest missing generation args in {path}: {missing_keys}."
        )
    for key, stored_value in stored_generation_args.items():
        if not hasattr(args, key):
            warn_and_raise(
                f"Cannot validate sharded manifest generation arg {key} in {path}: "
                "current CLI args do not expose this key."
            )
        current_value = _jsonable(getattr(args, key))
        if stored_value != current_value:
            warn_and_raise(
                f"Sharded manifest generation arg mismatch in {path}: "
                f"{key}={stored_value} != {current_value}."
            )
    return manifest


def _entry_consistency_tuple(entry):
    return (
        entry.get("sample_start"),
        entry.get("sample_end"),
        entry.get("saved_samples"),
        entry.get("snapshot_count"),
    )


def validate_stage_manifests(manifests):
    if not manifests:
        warn_and_raise("Cannot validate empty stage manifest set.")

    reference_name, reference = next(iter(manifests.items()))
    reference_entries = reference.get("entries", [])
    for name, manifest in list(manifests.items())[1:]:
        if manifest.get("total_saved_samples") != reference.get("total_saved_samples"):
            warn_and_raise(
                "Sharded manifest total_saved_samples mismatch: "
                f"{name}={manifest.get('total_saved_samples')} != "
                f"{reference_name}={reference.get('total_saved_samples')}."
            )
        if manifest.get("total_snapshots") != reference.get("total_snapshots"):
            warn_and_raise(
                "Sharded manifest total_snapshots mismatch: "
                f"{name}={manifest.get('total_snapshots')} != "
                f"{reference_name}={reference.get('total_snapshots')}."
            )
        entries = manifest.get("entries", [])
        if len(entries) != len(reference_entries):
            warn_and_raise(
                "Sharded manifest shard count mismatch: "
                f"{name}={len(entries)} != {reference_name}={len(reference_entries)}."
            )
        for idx, (entry, reference_entry) in enumerate(zip(entries, reference_entries)):
            if _entry_consistency_tuple(entry) != _entry_consistency_tuple(reference_entry):
                warn_and_raise(
                    "Sharded manifest entry mismatch: "
                    f"{name}[{idx}]={_entry_consistency_tuple(entry)} != "
                    f"{reference_name}[{idx}]={_entry_consistency_tuple(reference_entry)}."
                )
    return True


def successful_shard_seeds(args, raw_payload):
    seeds = build_global_seeds(args.dataset_seed, args.num_samples)
    sample_start = raw_payload["sample_start"]
    sample_end = raw_payload["sample_end"]
    shard_seeds = seeds[sample_start:sample_end]
    seed_mask = np.array(raw_payload["seed_mask"], dtype=bool)
    return shard_seeds[seed_mask]


def graph_map_id_bounds(dense_dataset):
    graph_map_id = dense_dataset[4]
    if len(graph_map_id) == 0:
        return None, None
    first = graph_map_id[0]
    last = graph_map_id[-1]
    if hasattr(first, "item"):
        first = first.item()
    if hasattr(last, "item"):
        last = last.item()
    return int(first), int(last) + 1
