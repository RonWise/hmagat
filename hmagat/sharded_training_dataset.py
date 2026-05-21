import bisect
import copy
import json
import pickle
import random
import time
from collections import OrderedDict

from loguru import logger
from torch.utils.data import Dataset, Sampler

from hmagat.downstream_shards import (
    iter_raw_expert_shards,
    load_stage_manifest,
    manifest_path,
    raw_expert_shards_dir,
    validate_stage_manifests,
    warn_and_raise,
)
from hmagat.imitation_dataset_pyg import MAPFGraphDataset, MAPFHypergraphDataset
from hmagat.progress_logging import ProgressLogger


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_int_list(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [int(value.item() if hasattr(value, "item") else value) for value in values]


def _requires_additional_data(additional_data_idx):
    return any(value is not None for value in additional_data_idx)


def _processed_manifest_args(args):
    if not getattr(args, "load_positions_separately", False):
        return args
    if not getattr(args, "use_edge_attr", False):
        return args

    processed_args = copy.copy(args)
    processed_args.use_edge_attr = False
    logger.info(
        "Validating processed shard manifest with use_edge_attr=False because "
        "training uses --load_positions_separately for edge attributes."
    )
    return processed_args


def load_training_manifests(
    args,
    *,
    use_hypergraphs,
    use_positions,
    additional_data_idx,
):
    manifests = {
        "processed_dataset": load_stage_manifest(
            _processed_manifest_args(args),
            "processed_dataset",
            expected_stage="processed",
        )
    }
    if use_hypergraphs:
        manifests["hypergraphs"] = load_stage_manifest(
            args, "hypergraphs", expected_stage="hypergraphs"
        )
    if _requires_additional_data(additional_data_idx):
        manifests["additional_data"] = load_stage_manifest(
            args, "additional_data", expected_stage="additional_data"
        )
    if use_positions:
        manifests["positions"] = load_stage_manifest(
            args, "positions", expected_stage="positions"
        )
    validate_stage_manifests(manifests)
    return manifests


def training_index_path(args):
    return manifest_path(_processed_manifest_args(args), "processed_dataset").parent / (
        "training_index.json"
    )


def sequence_audit_stamp_path(args):
    return manifest_path(_processed_manifest_args(args), "processed_dataset").parent / (
        "sequence_audit.json"
    )


def _processed_entry_signature(entry):
    return {
        "file_name": entry.get("file_name"),
        "shard_idx": entry.get("shard_idx"),
        "sample_start": entry.get("sample_start"),
        "sample_end": entry.get("sample_end"),
        "saved_samples": entry.get("saved_samples"),
        "snapshot_count": entry.get("snapshot_count"),
        "graph_map_id_start": entry.get("graph_map_id_start"),
        "graph_map_id_end": entry.get("graph_map_id_end"),
        "original_sample_ids": entry.get("original_sample_ids"),
    }


def _processed_manifest_signature(manifest):
    return [
        _processed_entry_signature(entry)
        for entry in manifest.get("entries", [])
    ]


def _sequence_audit_expected_metadata(args, processed_manifest):
    return {
        "version": 1,
        "audit_mode": "sharded_lightweight",
        "status": "passed",
        "num_samples": args.num_samples,
        "dataset_seed": args.dataset_seed,
        "override_name": args.override_name,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "total_saved_samples": processed_manifest.get("total_saved_samples"),
        "total_snapshots": processed_manifest.get("total_snapshots"),
    }


def build_sequence_audit_stamp_payload(
    args,
    processed_manifest,
    *,
    checked_train,
    checked_validation,
):
    payload = _sequence_audit_expected_metadata(args, processed_manifest)
    payload["checked_train"] = bool(checked_train)
    payload["checked_validation"] = bool(checked_validation)
    payload["processed_entries"] = _processed_manifest_signature(processed_manifest)
    return payload


def validate_sequence_audit_stamp_payload(payload, args, processed_manifest, path):
    expected = _sequence_audit_expected_metadata(args, processed_manifest)
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            warn_and_raise(
                "Sequence audit stamp metadata mismatch: "
                f"path={path}, {key}={payload.get(key)} != {expected_value}."
            )

    processed_entries = payload.get("processed_entries")
    expected_signature = _processed_manifest_signature(processed_manifest)
    if processed_entries != expected_signature:
        warn_and_raise(
            "Sequence audit stamp processed manifest signature mismatch: "
            f"path={path}."
        )

    if not payload.get("checked_train") and not payload.get("checked_validation"):
        warn_and_raise(
            "Sequence audit stamp indicates no train or validation split was "
            f"checked: path={path}."
        )


def write_sequence_audit_stamp(
    args,
    processed_manifest,
    *,
    checked_train,
    checked_validation,
):
    path = sequence_audit_stamp_path(args)
    payload = build_sequence_audit_stamp_payload(
        args,
        processed_manifest,
        checked_train=checked_train,
        checked_validation=checked_validation,
    )
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path, payload


def load_sequence_audit_stamp(args, processed_manifest=None):
    if processed_manifest is None:
        processed_manifest = load_stage_manifest(
            _processed_manifest_args(args),
            "processed_dataset",
            expected_stage="processed",
        )
    path = sequence_audit_stamp_path(args)
    with open(path) as f:
        payload = json.load(f)
    validate_sequence_audit_stamp_payload(payload, args, processed_manifest, path)
    return payload


def _validate_training_index_payload(index_payload, manifests, args, path):
    processed_manifest = manifests["processed_dataset"]
    expected = {
        "version": 1,
        "num_samples": args.num_samples,
        "dataset_seed": args.dataset_seed,
        "override_name": args.override_name,
        "total_saved_samples": processed_manifest.get("total_saved_samples"),
        "total_snapshots": processed_manifest.get("total_snapshots"),
    }
    for key, expected_value in expected.items():
        if index_payload.get(key) != expected_value:
            warn_and_raise(
                "Sharded training index metadata mismatch: "
                f"path={path}, {key}={index_payload.get(key)} != {expected_value}."
            )

    expected_signature = _processed_manifest_signature(processed_manifest)
    if index_payload.get("processed_entries") != expected_signature:
        warn_and_raise(
            "Sharded training index processed manifest signature mismatch: "
            f"path={path}."
        )

    episodes = index_payload.get("episodes")
    shards = index_payload.get("shards")
    if not isinstance(episodes, list) or not isinstance(shards, list):
        warn_and_raise(f"Invalid sharded training index payload: path={path}.")
    if len(episodes) != processed_manifest.get("total_saved_samples"):
        warn_and_raise(
            "Sharded training index episode count mismatch: "
            f"path={path}, episodes={len(episodes)} != "
            f"{processed_manifest.get('total_saved_samples')}."
        )
    indexed_snapshots = sum(
        int(episode["global_end"]) - int(episode["global_start"])
        for episode in episodes
    )
    if indexed_snapshots != processed_manifest.get("total_snapshots"):
        warn_and_raise(
            "Sharded training index snapshot count mismatch: "
            f"path={path}, snapshots={indexed_snapshots} != "
            f"{processed_manifest.get('total_snapshots')}."
        )
    for episode_idx, episode in enumerate(episodes):
        if "original_sample_id" not in episode:
            warn_and_raise(
                "Sharded training index episode is missing original_sample_id: "
                f"path={path}, episode_index={episode_idx}."
            )


def _episode_ranges_from_map_ids(
    map_ids,
    shard_idx,
    snapshot_start,
    seen_episode_ids,
    original_sample_ids,
):
    episodes = []
    if not map_ids:
        if original_sample_ids:
            warn_and_raise(
                "Sharded training index received original_sample_ids for an empty "
                f"processed shard: shard={shard_idx}."
            )
        return episodes

    if len(original_sample_ids) == 0:
        warn_and_raise(
            "Sharded training index requires non-empty original_sample_ids for "
            f"non-empty processed shard {shard_idx}."
        )

    current_episode_id = map_ids[0]
    current_episode_position = 0
    current_original_sample_id = int(original_sample_ids[current_episode_position])
    local_start = 0
    if current_episode_id in seen_episode_ids:
        warn_and_raise(
            "Sharded training index cannot handle episodes spanning multiple "
            f"processed shards: episode_id={current_episode_id}, shard={shard_idx}."
        )
    seen_episode_ids.add(current_episode_id)

    for local_idx, episode_id in enumerate(map_ids[1:], start=1):
        if episode_id == current_episode_id:
            continue
        if episode_id in seen_episode_ids:
            warn_and_raise(
                "Sharded training index encountered non-contiguous episode ids: "
                f"episode_id={episode_id}, shard={shard_idx}."
            )
        episodes.append(
            {
                "episode_id": int(current_episode_id),
                "original_sample_id": int(current_original_sample_id),
                "global_start": int(snapshot_start + local_start),
                "global_end": int(snapshot_start + local_idx),
                "shard_idx": int(shard_idx),
            }
        )
        current_episode_position += 1
        if current_episode_position >= len(original_sample_ids):
            warn_and_raise(
                "Sharded training index found more processed episodes than "
                "original sample ids in shard metadata: "
                f"shard={shard_idx}, episode_position={current_episode_position}."
            )
        current_episode_id = episode_id
        current_original_sample_id = int(original_sample_ids[current_episode_position])
        local_start = local_idx
        seen_episode_ids.add(current_episode_id)

    episodes.append(
        {
            "episode_id": int(current_episode_id),
            "original_sample_id": int(current_original_sample_id),
            "global_start": int(snapshot_start + local_start),
            "global_end": int(snapshot_start + len(map_ids)),
            "shard_idx": int(shard_idx),
        }
    )
    if current_episode_position + 1 != len(original_sample_ids):
        warn_and_raise(
            "Sharded training index original_sample_ids length mismatch: "
            f"shard={shard_idx}, episodes={current_episode_position + 1}, "
            f"original_sample_ids={len(original_sample_ids)}."
        )
    return episodes


def _raw_shard_original_sample_ids_by_range(args):
    mapping = {}
    for record in iter_raw_expert_shards(args):
        payload = record["payload"]
        key = (int(payload["sample_start"]), int(payload["sample_end"]))
        if key in mapping:
            warn_and_raise(
                "Duplicate raw expert shard range while recovering original "
                f"sample ids: range={key}."
            )
        mapping[key] = [
            int(payload["sample_start"] + local_idx)
            for local_idx, success in enumerate(payload["seed_mask"])
            if success
        ]
    return mapping


def _infer_original_sample_ids_from_graph_map_range(entry):
    graph_map_id_start = entry.get("graph_map_id_start")
    graph_map_id_end = entry.get("graph_map_id_end")
    if graph_map_id_start is None or graph_map_id_end is None:
        return None
    inferred = list(range(int(graph_map_id_start), int(graph_map_id_end)))
    if len(inferred) != int(entry["saved_samples"]):
        return None
    return inferred


def _build_training_index_payload(manifests, args):
    processed_manifest = manifests["processed_dataset"]
    entries = processed_manifest["entries"]
    progress = ProgressLogger(
        "Building sharded training sidecar index",
        len(entries),
        every_n=1,
        every_seconds=30.0,
    )
    snapshot_start = 0
    shards = []
    episodes = []
    seen_episode_ids = set()
    raw_original_sample_ids_by_range = None
    for shard_idx, entry in enumerate(entries):
        logger.info(
            "Indexing processed shard for training sidecar: "
            f"shard={shard_idx}, path={entry['path']}, "
            f"manifest_snapshots={entry['snapshot_count']}"
        )
        dense_dataset = _load_pickle(entry["path"])
        map_ids = _to_int_list(dense_dataset[4])
        snapshot_count = int(entry["snapshot_count"])
        if len(map_ids) != snapshot_count:
            warn_and_raise(
                "Processed shard graph_map_id length mismatch while building "
                "training index: "
                f"path={entry['path']}, graph_map_id={len(map_ids)}, "
                f"manifest={snapshot_count}."
            )

        original_sample_ids = entry.get("original_sample_ids")
        if original_sample_ids is None:
            key = (int(entry["sample_start"]), int(entry["sample_end"]))
            inferred_original_sample_ids = _infer_original_sample_ids_from_graph_map_range(
                entry
            )
            raw_shards_present = any(raw_expert_shards_dir(args).glob("*.pkl"))
            if raw_shards_present:
                if raw_original_sample_ids_by_range is None:
                    logger.warning(
                        "Processed shard manifest lacks original_sample_ids; "
                        "recovering them from raw expert shard metadata."
                    )
                    raw_original_sample_ids_by_range = (
                        _raw_shard_original_sample_ids_by_range(args)
                    )
                original_sample_ids = raw_original_sample_ids_by_range.get(key)
                if original_sample_ids is None:
                    warn_and_raise(
                        "Cannot recover original_sample_ids for processed shard: "
                        f"shard={shard_idx}, range={key}."
                    )
            elif inferred_original_sample_ids is not None:
                logger.warning(
                    "Processed shard manifest lacks original_sample_ids and raw "
                    "expert shards are unavailable; inferring them from contiguous "
                    f"graph_map_id range for shard={shard_idx}, range={key}. "
                    "This fallback is only correct when processed episode ids still "
                    "match original sample ids."
                )
                original_sample_ids = inferred_original_sample_ids
            else:
                warn_and_raise(
                    "Processed shard manifest lacks original_sample_ids, raw expert "
                    "shards are unavailable, and graph_map_id range is insufficient "
                    f"to infer them: shard={shard_idx}, range={key}."
                )
        original_sample_ids = [int(value) for value in original_sample_ids]
        if len(original_sample_ids) != int(entry["saved_samples"]):
            warn_and_raise(
                "Processed shard original_sample_ids length mismatch: "
                f"shard={shard_idx}, original_sample_ids={len(original_sample_ids)} "
                f"!= saved_samples={entry['saved_samples']}."
            )

        snapshot_end = snapshot_start + snapshot_count
        shards.append(
            {
                "shard_idx": int(shard_idx),
                "snapshot_start": int(snapshot_start),
                "snapshot_end": int(snapshot_end),
            }
        )
        episodes.extend(
            _episode_ranges_from_map_ids(
                map_ids,
                shard_idx,
                snapshot_start,
                seen_episode_ids,
                original_sample_ids,
            )
        )
        snapshot_start = snapshot_end
        progress.update(
            shard_idx + 1,
            extra=f"indexed_snapshots={snapshot_start}",
        )

    return {
        "version": 1,
        "num_samples": args.num_samples,
        "dataset_seed": args.dataset_seed,
        "override_name": args.override_name,
        "total_saved_samples": processed_manifest["total_saved_samples"],
        "total_snapshots": processed_manifest["total_snapshots"],
        "processed_entries": _processed_manifest_signature(processed_manifest),
        "shards": shards,
        "episodes": episodes,
    }


class ShardedTrainingIndex:
    def __init__(self, manifests, payload):
        self.manifests = manifests
        self.payload = payload
        self.total_episodes = int(payload["total_saved_samples"])

    @classmethod
    def load_or_build(cls, args, manifests):
        path = training_index_path(args)
        if path.exists():
            with open(path) as f:
                payload = json.load(f)
            _validate_training_index_payload(payload, manifests, args, path)
            logger.info(f"Loaded sharded training sidecar index: path={path}")
            return cls(manifests, payload)

        logger.warning(
            "Sharded training sidecar index is missing; rebuilding it from "
            f"processed shards: path={path}. This is an explicit logged fallback."
        )
        payload = _build_training_index_payload(manifests, args)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        logger.info(
            "Sharded training sidecar index written: "
            f"path={path}, episodes={len(payload['episodes'])}, "
            f"snapshots={payload['total_snapshots']}"
        )
        return cls(manifests, payload)

    def _entry_for(self, manifest_name, shard_idx):
        manifest = self.manifests.get(manifest_name)
        if manifest is None:
            return None
        return manifest["entries"][shard_idx]

    def shards_for_training(self):
        processed_entries = self.manifests["processed_dataset"]["entries"]
        shards = []
        shard_starts = []
        for shard in self.payload["shards"]:
            shard_idx = int(shard["shard_idx"])
            shards.append(
                {
                    "shard_idx": shard_idx,
                    "snapshot_start": int(shard["snapshot_start"]),
                    "snapshot_end": int(shard["snapshot_end"]),
                    "processed": processed_entries[shard_idx],
                    "hypergraphs": self._entry_for("hypergraphs", shard_idx),
                    "additional_data": self._entry_for("additional_data", shard_idx),
                    "positions": self._entry_for("positions", shard_idx),
                }
            )
            shard_starts.append(int(shard["snapshot_start"]))
        return shards, shard_starts

    def split(self, episode_start, episode_end):
        global_indices = []
        graph_map_id = []
        episode_ids = []
        episode_shard_indices = []
        episode_original_sample_ids = []
        selected_episodes = []
        shard_snapshot_starts = {
            int(shard["shard_idx"]): int(shard["snapshot_start"])
            for shard in self.payload["shards"]
        }
        for episode in self.payload["episodes"]:
            episode_id = int(episode["episode_id"])
            if not episode_start <= episode_id < episode_end:
                continue
            global_start = int(episode["global_start"])
            global_end = int(episode["global_end"])
            shard_idx = int(episode["shard_idx"])
            shard_start = shard_snapshot_starts[shard_idx]
            global_indices.extend(range(global_start, global_end))
            graph_map_id.extend([episode_id] * (global_end - global_start))
            episode_ids.append(episode_id)
            episode_shard_indices.append(shard_idx)
            original_sample_id = int(episode["original_sample_id"])
            episode_original_sample_ids.append(original_sample_id)
            selected_episodes.append(
                {
                    "episode_id": episode_id,
                    "original_sample_id": original_sample_id,
                    "shard_idx": shard_idx,
                    "global_start": global_start,
                    "global_end": global_end,
                    "local_start": int(global_start - shard_start),
                    "local_end": int(global_end - shard_start),
                    "length": int(global_end - global_start),
                }
            )
        return {
            "global_indices": global_indices,
            "graph_map_id": graph_map_id,
            "episode_ids": episode_ids,
            "episode_shard_indices": episode_shard_indices,
            "episode_original_sample_ids": episode_original_sample_ids,
            "episodes": selected_episodes,
        }


class _ShardedDatasetMixin:
    def _init_sharded_runtime_stats(self):
        self._cache_size = int(getattr(self.args, "sharded_dataset_cache_size", 1))
        self._cached_datasets = OrderedDict()
        self._runtime_context = {
            "dataset_role": None,
            "phase": None,
            "epoch": None,
            "batch_idx": None,
        }
        self.reset_runtime_stats()

    def reset_runtime_stats(self):
        self._runtime_stats = {
            "shard_accesses": 0,
            "shard_loads": 0,
            "shard_switches": 0,
            "unique_shards_touched": set(),
            "shard_load_time_sec": 0.0,
            "processed_payload_load_sec": 0.0,
            "positions_payload_load_sec": 0.0,
            "additional_data_payload_load_sec": 0.0,
            "hypergraph_payload_load_sec": 0.0,
            "shard_dataset_materialization_sec": 0.0,
            "cache_hits": 0,
            "cache_evictions": 0,
            "load_events_recorded": 0,
        }
        self._last_accessed_shard_idx = None
        self._recent_shard_events = []

    def set_runtime_context(
        self,
        *,
        dataset_role=None,
        phase=None,
        epoch=None,
        batch_idx=None,
    ):
        if dataset_role is not None:
            self._runtime_context["dataset_role"] = str(dataset_role)
        if phase is not None:
            self._runtime_context["phase"] = str(phase)
        if epoch is not None:
            self._runtime_context["epoch"] = int(epoch)
        if batch_idx is not None:
            self._runtime_context["batch_idx"] = int(batch_idx)

    def clear_runtime_batch_context(self):
        self._runtime_context["batch_idx"] = None

    def _record_shard_event(self, *, event_type, shard_idx, caller):
        context = dict(self._runtime_context)
        event = {
            "event_type": str(event_type),
            "shard_idx": int(shard_idx),
            "caller": None if caller is None else str(caller),
            "dataset_role": context.get("dataset_role"),
            "phase": context.get("phase"),
            "epoch": context.get("epoch"),
            "batch_idx": context.get("batch_idx"),
        }
        self._recent_shard_events.append(event)
        if len(self._recent_shard_events) > 16:
            self._recent_shard_events.pop(0)
        self._runtime_stats["load_events_recorded"] += 1
        return event

    @staticmethod
    def _format_shard_event(event):
        parts = [
            str(event["event_type"]),
            f"shard={event['shard_idx']}",
        ]
        if event.get("caller") is not None:
            parts.append(f"caller={event['caller']}")
        if event.get("dataset_role") is not None:
            parts.append(f"dataset_role={event['dataset_role']}")
        if event.get("phase") is not None:
            parts.append(f"phase={event['phase']}")
        if event.get("epoch") is not None:
            parts.append(f"epoch={event['epoch']}")
        if event.get("batch_idx") is not None:
            parts.append(f"batch_idx={event['batch_idx']}")
        return ", ".join(parts)

    def runtime_stats_snapshot(self):
        stats = dict(self._runtime_stats)
        stats["unique_shards_touched"] = int(len(stats["unique_shards_touched"]))
        stats["cache_size"] = int(self._cache_size)
        stats["dataset_role"] = self._runtime_context.get("dataset_role")
        stats["phase"] = self._runtime_context.get("phase")
        stats["epoch"] = self._runtime_context.get("epoch")
        stats["batch_idx"] = self._runtime_context.get("batch_idx")
        stats["recent_shard_events"] = [
            self._format_shard_event(event) for event in self._recent_shard_events
        ]
        return stats

    def _load_shard_dataset(self, shard_idx, *, caller=None):
        self._runtime_stats["shard_accesses"] += 1
        self._runtime_stats["unique_shards_touched"].add(int(shard_idx))
        if (
            self._last_accessed_shard_idx is not None
            and self._last_accessed_shard_idx != shard_idx
        ):
            self._runtime_stats["shard_switches"] += 1
        self._last_accessed_shard_idx = shard_idx

        cached_dataset = self._cached_datasets.get(int(shard_idx))
        if cached_dataset is not None:
            self._runtime_stats["cache_hits"] += 1
            self._cached_datasets.move_to_end(int(shard_idx))
            self._record_shard_event(
                event_type="cache_hit", shard_idx=shard_idx, caller=caller
            )
            return cached_dataset

        start_time = time.monotonic()
        shard_event = self._record_shard_event(
            event_type="cache_miss_load", shard_idx=shard_idx, caller=caller
        )
        shard = self.shards[shard_idx]
        logger.info(
            "Loading sharded training artifact bundle: "
            f"shard={shard_idx}, "
            f"snapshot_range=[{shard['snapshot_start']}, {shard['snapshot_end']}), "
            f"processed_snapshots={shard['processed'].get('snapshot_count')}, "
            f"uses_hypergraphs={self.use_hypergraphs}, "
            f"uses_positions={self.use_positions}, "
            "uses_additional_data="
            f"{_requires_additional_data(self.additional_data_idx)}, "
            f"context=({self._format_shard_event(shard_event)})"
        )
        payload_load_start_time = time.monotonic()
        dense_dataset = _load_pickle(shard["processed"]["path"])
        self._runtime_stats["processed_payload_load_sec"] += (
            time.monotonic() - payload_load_start_time
        )
        logger.info(
            "Processed shard payload loaded: "
            f"shard={shard_idx}, path={shard['processed']['path']}"
        )
        if self.use_positions:
            if shard["positions"] is None:
                warn_and_raise(
                    "Training requested separately loaded positions, but no "
                    f"positions shard is available for shard {shard_idx}."
                )
            payload_load_start_time = time.monotonic()
            positions = _load_pickle(shard["positions"]["path"])
            self._runtime_stats["positions_payload_load_sec"] += (
                time.monotonic() - payload_load_start_time
            )
            dense_dataset = (*dense_dataset, positions)
            logger.info(
                "Positions shard payload loaded: "
                f"shard={shard_idx}, path={shard['positions']['path']}"
            )

        additional_data = None
        if _requires_additional_data(self.additional_data_idx):
            if shard["additional_data"] is None:
                warn_and_raise(
                    "Training requested additional data, but no additional_data "
                    f"shard is available for shard {shard_idx}."
                )
            payload_load_start_time = time.monotonic()
            additional_data = _load_pickle(shard["additional_data"]["path"])
            self._runtime_stats["additional_data_payload_load_sec"] += (
                time.monotonic() - payload_load_start_time
            )
            logger.info(
                "Additional-data shard payload loaded: "
                f"shard={shard_idx}, path={shard['additional_data']['path']}"
            )

        if self.use_hypergraphs:
            if shard["hypergraphs"] is None:
                warn_and_raise(
                    "Training requested hypergraphs, but no hypergraph shard is "
                    f"available for shard {shard_idx}."
                )
            payload_load_start_time = time.monotonic()
            hypergraphs = _load_pickle(shard["hypergraphs"]["path"])
            self._runtime_stats["hypergraph_payload_load_sec"] += (
                time.monotonic() - payload_load_start_time
            )
            logger.info(
                "Hypergraph shard payload loaded: "
                f"shard={shard_idx}, path={shard['hypergraphs']['path']}"
            )
            dataset_build_start_time = time.monotonic()
            dataset = MAPFHypergraphDataset(
                dense_dataset,
                hypergraphs,
                additional_data=additional_data,
                **self.dataset_kwargs,
            )
        else:
            dataset_build_start_time = time.monotonic()
            dataset = MAPFGraphDataset(
                dense_dataset,
                additional_data=additional_data,
                **self.dataset_kwargs,
            )
        self._runtime_stats["shard_dataset_materialization_sec"] += (
            time.monotonic() - dataset_build_start_time
        )

        logger.info(
            "Sharded training artifact bundle ready: "
            f"shard={shard_idx}, dataset_len={len(dataset)}"
        )

        self._cached_datasets[int(shard_idx)] = dataset
        self._cached_datasets.move_to_end(int(shard_idx))
        while len(self._cached_datasets) > self._cache_size:
            evicted_shard_idx, _ = self._cached_datasets.popitem(last=False)
            self._runtime_stats["cache_evictions"] += 1
            evict_event = self._record_shard_event(
                event_type="cache_evict", shard_idx=evicted_shard_idx, caller=caller
            )
            logger.info(
                "Evicted sharded training artifact bundle from cache: {}",
                self._format_shard_event(evict_event),
            )
        self._runtime_stats["shard_loads"] += 1
        self._runtime_stats["shard_load_time_sec"] += time.monotonic() - start_time
        return dataset


class ShardedSnapshotDataset(_ShardedDatasetMixin, Dataset):
    def __init__(
        self,
        args,
        *,
        episode_start,
        episode_end,
        use_hypergraphs,
        use_edge_attr,
        edge_attr_opts="straight",
        target_vec=None,
        use_target_vec=None,
        additional_data_idx=None,
        use_edge_attr_for_messages=None,
        allow_empty=False,
        training_index=None,
    ):
        if additional_data_idx is None:
            additional_data_idx = [None, None, None]

        self.args = args
        self.episode_start = int(episode_start)
        self.episode_end = int(episode_end)
        self.use_hypergraphs = use_hypergraphs
        self.allow_empty = allow_empty
        self.use_edge_attr = use_edge_attr
        self.use_positions = bool(
            use_edge_attr and getattr(args, "load_positions_separately", False)
        )
        self.dataset_kwargs = dict(
            use_edge_attr=use_edge_attr,
            edge_attr_opts=edge_attr_opts,
            target_vec=target_vec,
            use_target_vec=use_target_vec,
            additional_data_idx=additional_data_idx,
            use_edge_attr_for_messages=use_edge_attr_for_messages,
        )
        self.additional_data_idx = additional_data_idx
        if training_index is None:
            manifests = load_training_manifests(
                args,
                use_hypergraphs=use_hypergraphs,
                use_positions=self.use_positions,
                additional_data_idx=additional_data_idx,
            )
            training_index = ShardedTrainingIndex.load_or_build(args, manifests)
        self.training_index = training_index
        self.manifests = training_index.manifests

        self.shards = []
        self.shard_starts = []
        self.global_indices = []
        self.graph_map_id = []
        self.episode_ids = []
        self.episode_shard_indices = []
        self.episode_original_sample_ids = []
        self._init_sharded_runtime_stats()
        self._build_index()

        logger.info(
            "Sharded training dataset ready: "
            f"episodes=[{self.episode_start}, {self.episode_end}), "
            f"snapshots={len(self.global_indices)}, shards={len(self.shards)}, "
            f"use_hypergraphs={self.use_hypergraphs}, "
            f"use_positions={self.use_positions}, "
            f"uses_additional_data={_requires_additional_data(self.additional_data_idx)}"
        )

    def _build_index(self):
        self.shards, self.shard_starts = self.training_index.shards_for_training()
        split = self.training_index.split(self.episode_start, self.episode_end)
        self.global_indices = split["global_indices"]
        self.graph_map_id = split["graph_map_id"]
        self.episode_ids = split["episode_ids"]
        self.episode_shard_indices = split["episode_shard_indices"]
        self.episode_original_sample_ids = split["episode_original_sample_ids"]
        if not self.global_indices:
            if self.allow_empty:
                logger.info(
                    "Sharded training dataset split is empty and explicitly allowed: "
                    f"episodes=[{self.episode_start}, {self.episode_end})."
                )
                return
            warn_and_raise(
                "Sharded training dataset split is empty: "
                f"episodes=[{self.episode_start}, {self.episode_end})."
            )
        logger.info(
            "Using sharded training sidecar index split: "
            f"episodes=[{self.episode_start}, {self.episode_end}), "
            f"selected_episodes={len(self.episode_ids)}, "
            f"selected_snapshots={len(self.global_indices)}"
        )

    def _entry_for(self, manifest_name, shard_idx):
        manifest = self.manifests.get(manifest_name)
        if manifest is None:
            return None
        return manifest["entries"][shard_idx]

    def __len__(self):
        return len(self.global_indices)

    def _shard_for_global_index(self, global_index):
        shard_idx = bisect.bisect_right(self.shard_starts, global_index) - 1
        if shard_idx < 0 or shard_idx >= len(self.shards):
            warn_and_raise(f"Global snapshot index is outside sharded dataset: {global_index}.")
        shard = self.shards[shard_idx]
        if not shard["snapshot_start"] <= global_index < shard["snapshot_end"]:
            warn_and_raise(
                "Global snapshot index does not belong to resolved shard: "
                f"index={global_index}, shard={shard_idx}, "
                f"range=[{shard['snapshot_start']}, {shard['snapshot_end']})."
            )
        return shard_idx, global_index - shard["snapshot_start"]

    def __getitem__(self, index):
        global_index = self.global_indices[index]
        shard_idx, local_index = self._shard_for_global_index(global_index)
        return self._load_shard_dataset(
            shard_idx, caller="snapshot_getitem"
        )[local_index]


class ShardedSequenceEpisode:
    __slots__ = ("_sequence_dataset", "shard_idx", "local_start", "local_end")

    def __init__(self, sequence_dataset, *, shard_idx, local_start, local_end):
        self._sequence_dataset = sequence_dataset
        self.shard_idx = int(shard_idx)
        self.local_start = int(local_start)
        self.local_end = int(local_end)

    def __len__(self):
        return self.local_end - self.local_start

    def _validate_index(self, index):
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(
                f"ShardedSequenceEpisode index {index} is out of range for length {length}."
            )
        return index

    def __getitem__(self, index):
        index = self._validate_index(index)
        shard_dataset = self._sequence_dataset._load_shard_dataset(
            self.shard_idx, caller="sequence_episode_getitem"
        )
        return shard_dataset[self.local_start + index]

    def __iter__(self):
        shard_dataset = self._sequence_dataset._load_shard_dataset(
            self.shard_idx, caller="sequence_episode_iter"
        )
        for local_index in range(self.local_start, self.local_end):
            yield shard_dataset[local_index]

    def _append_timestep_items(
        self, timestep_items, timestep_active_indices, episode_idx
    ):
        shard_dataset = self._sequence_dataset._load_shard_dataset(
            self.shard_idx, caller="sequence_episode_append_timestep_items"
        )
        for timestep, local_index in enumerate(range(self.local_start, self.local_end)):
            timestep_items[timestep].append(shard_dataset[local_index])
            timestep_active_indices[timestep].append(int(episode_idx))


class ShardedSequenceDataset(_ShardedDatasetMixin, Dataset):
    def __init__(
        self,
        args,
        *,
        episode_start,
        episode_end,
        use_hypergraphs,
        use_edge_attr,
        edge_attr_opts="straight",
        target_vec=None,
        use_target_vec=None,
        additional_data_idx=None,
        use_edge_attr_for_messages=None,
        allow_empty=False,
        training_index=None,
    ):
        if additional_data_idx is None:
            additional_data_idx = [None, None, None]

        self.args = args
        self.episode_start = int(episode_start)
        self.episode_end = int(episode_end)
        self.use_hypergraphs = use_hypergraphs
        self.allow_empty = allow_empty
        self.use_edge_attr = use_edge_attr
        self.use_positions = bool(
            use_edge_attr and getattr(args, "load_positions_separately", False)
        )
        self.dataset_kwargs = dict(
            use_edge_attr=use_edge_attr,
            edge_attr_opts=edge_attr_opts,
            target_vec=target_vec,
            use_target_vec=use_target_vec,
            additional_data_idx=additional_data_idx,
            use_edge_attr_for_messages=use_edge_attr_for_messages,
        )
        self.additional_data_idx = additional_data_idx
        if training_index is None:
            manifests = load_training_manifests(
                args,
                use_hypergraphs=use_hypergraphs,
                use_positions=self.use_positions,
                additional_data_idx=additional_data_idx,
            )
            training_index = ShardedTrainingIndex.load_or_build(args, manifests)
        self.training_index = training_index
        self.manifests = training_index.manifests

        self.shards = []
        self.shard_starts = []
        self.episodes = []
        self.episode_ids = []
        self.episode_shard_indices = []
        self.episode_original_sample_ids = []
        self._init_sharded_runtime_stats()
        self._build_index()

        logger.info(
            "Sharded sequence dataset ready: "
            f"episodes=[{self.episode_start}, {self.episode_end}), "
            f"selected_episodes={len(self.episodes)}, shards={len(self.shards)}, "
            f"use_hypergraphs={self.use_hypergraphs}, "
            f"use_positions={self.use_positions}, "
            f"uses_additional_data={_requires_additional_data(self.additional_data_idx)}"
        )

    @classmethod
    def from_snapshot_dataset(cls, snapshot_dataset):
        return cls(
            snapshot_dataset.args,
            episode_start=snapshot_dataset.episode_start,
            episode_end=snapshot_dataset.episode_end,
            use_hypergraphs=snapshot_dataset.use_hypergraphs,
            use_edge_attr=snapshot_dataset.use_edge_attr,
            edge_attr_opts=snapshot_dataset.dataset_kwargs.get("edge_attr_opts", "straight"),
            target_vec=snapshot_dataset.dataset_kwargs.get("target_vec"),
            use_target_vec=snapshot_dataset.dataset_kwargs.get("use_target_vec"),
            additional_data_idx=snapshot_dataset.additional_data_idx,
            use_edge_attr_for_messages=snapshot_dataset.dataset_kwargs.get(
                "use_edge_attr_for_messages"
            ),
            allow_empty=snapshot_dataset.allow_empty,
            training_index=snapshot_dataset.training_index,
        )

    def _build_index(self):
        self.shards, self.shard_starts = self.training_index.shards_for_training()
        split = self.training_index.split(self.episode_start, self.episode_end)
        self.episodes = split["episodes"]
        self.episode_ids = split["episode_ids"]
        self.episode_shard_indices = split["episode_shard_indices"]
        self.episode_original_sample_ids = split["episode_original_sample_ids"]
        if not self.episodes:
            if self.allow_empty:
                logger.info(
                    "Sharded sequence dataset split is empty and explicitly allowed: "
                    f"episodes=[{self.episode_start}, {self.episode_end})."
                )
                return
            warn_and_raise(
                "Sharded sequence dataset split is empty: "
                f"episodes=[{self.episode_start}, {self.episode_end})."
            )
        logger.info(
            "Using sharded sequence sidecar index split: "
            f"episodes=[{self.episode_start}, {self.episode_end}), "
            f"selected_episodes={len(self.episodes)}"
        )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        episode = self.episodes[index]
        return ShardedSequenceEpisode(
            self,
            shard_idx=int(episode["shard_idx"]),
            local_start=int(episode["local_start"]),
            local_end=int(episode["local_end"]),
        )


class ShardedEpisodeSampler(Sampler):
    def __init__(self, snapshot_dataset, *, shuffle=True):
        if not hasattr(snapshot_dataset, "episode_shard_indices"):
            warn_and_raise(
                "Cannot build ShardedEpisodeSampler: snapshot dataset does not "
                "expose episode_shard_indices."
            )
        self.episode_shard_indices = list(snapshot_dataset.episode_shard_indices)
        self.shuffle = shuffle
        self.sequence_indices = list(range(len(self.episode_shard_indices)))

    def __iter__(self):
        indices = list(self.sequence_indices)
        if self.shuffle:
            random.shuffle(indices)
        yield from indices

    def __len__(self):
        return len(self.episode_shard_indices)


class ShardedEpisodeBatchSampler(Sampler):
    def __init__(self, episode_dataset, *, batch_size, shuffle=True, drop_last=False):
        if batch_size <= 0:
            warn_and_raise(
                f"ShardedEpisodeBatchSampler requires positive batch_size, got {batch_size}."
            )
        if not hasattr(episode_dataset, "episode_shard_indices"):
            warn_and_raise(
                "Cannot build ShardedEpisodeBatchSampler: dataset does not expose "
                "episode_shard_indices."
            )
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.episode_shard_indices = list(episode_dataset.episode_shard_indices)

    def _grouped_indices(self):
        grouped = {}
        for episode_index, shard_idx in enumerate(self.episode_shard_indices):
            grouped.setdefault(int(shard_idx), []).append(int(episode_index))
        return grouped

    def __iter__(self):
        grouped = self._grouped_indices()
        shard_order = sorted(grouped)
        if self.shuffle:
            random.shuffle(shard_order)
        for shard_idx in shard_order:
            episode_indices = list(grouped[shard_idx])
            if self.shuffle:
                random.shuffle(episode_indices)
            for start in range(0, len(episode_indices), self.batch_size):
                batch = episode_indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        total = 0
        grouped = self._grouped_indices()
        for episode_indices in grouped.values():
            full_batches, remainder = divmod(len(episode_indices), self.batch_size)
            total += full_batches
            if remainder and not self.drop_last:
                total += 1
        return total


def build_sharded_snapshot_datasets(
    args,
    *,
    hypergraph_model,
    additional_data_idx,
    dataset_kwargs,
):
    manifests = load_training_manifests(
        args,
        use_hypergraphs=hypergraph_model,
        use_positions=bool(
            dataset_kwargs.get("use_edge_attr")
            and getattr(args, "load_positions_separately", False)
        ),
        additional_data_idx=additional_data_idx,
    )
    total_episodes = int(manifests["processed_dataset"]["total_saved_samples"])
    train_end = int(
        total_episodes * (1 - args.validation_fraction - args.test_fraction)
    )
    validation_end = train_end + int(total_episodes * args.validation_fraction)
    train_end = min(train_end, total_episodes)
    validation_end = min(validation_end, total_episodes)

    logger.info(
        "Using sharded training split: "
        f"episodes={total_episodes}, train=[0, {train_end}), "
        f"validation=[{train_end}, {validation_end})"
    )
    training_index = ShardedTrainingIndex.load_or_build(args, manifests)
    allow_empty_validation = bool(args.skip_validation and validation_end <= train_end)
    train_dataset = ShardedSnapshotDataset(
        args,
        episode_start=0,
        episode_end=train_end,
        use_hypergraphs=hypergraph_model,
        training_index=training_index,
        **dataset_kwargs,
    )
    validation_dataset = ShardedSnapshotDataset(
        args,
        episode_start=train_end,
        episode_end=validation_end,
        use_hypergraphs=hypergraph_model,
        allow_empty=allow_empty_validation,
        training_index=training_index,
        **dataset_kwargs,
    )
    return train_dataset, validation_dataset, train_end, validation_end
