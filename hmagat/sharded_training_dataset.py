import bisect
import copy
import json
import pickle
import random

from loguru import logger
from torch.utils.data import Dataset, Sampler

from hmagat.downstream_shards import (
    load_stage_manifest,
    manifest_path,
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
    }


def _processed_manifest_signature(manifest):
    return [
        _processed_entry_signature(entry)
        for entry in manifest.get("entries", [])
    ]


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


def _episode_ranges_from_map_ids(map_ids, shard_idx, snapshot_start, seen_episode_ids):
    episodes = []
    if not map_ids:
        return episodes

    current_episode_id = map_ids[0]
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
                "global_start": int(snapshot_start + local_start),
                "global_end": int(snapshot_start + local_idx),
                "shard_idx": int(shard_idx),
            }
        )
        current_episode_id = episode_id
        local_start = local_idx
        seen_episode_ids.add(current_episode_id)

    episodes.append(
        {
            "episode_id": int(current_episode_id),
            "global_start": int(snapshot_start + local_start),
            "global_end": int(snapshot_start + len(map_ids)),
            "shard_idx": int(shard_idx),
        }
    )
    return episodes


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
        for episode in self.payload["episodes"]:
            episode_id = int(episode["episode_id"])
            if not episode_start <= episode_id < episode_end:
                continue
            global_start = int(episode["global_start"])
            global_end = int(episode["global_end"])
            global_indices.extend(range(global_start, global_end))
            graph_map_id.extend([episode_id] * (global_end - global_start))
            episode_ids.append(episode_id)
            episode_shard_indices.append(int(episode["shard_idx"]))
        return {
            "global_indices": global_indices,
            "graph_map_id": graph_map_id,
            "episode_ids": episode_ids,
            "episode_shard_indices": episode_shard_indices,
        }


class ShardedSnapshotDataset(Dataset):
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
        self._cached_shard_idx = None
        self._cached_dataset = None
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

    def _load_shard_dataset(self, shard_idx):
        if self._cached_shard_idx == shard_idx:
            return self._cached_dataset

        shard = self.shards[shard_idx]
        logger.info(
            "Loading sharded training artifact bundle: "
            f"shard={shard_idx}, "
            f"snapshot_range=[{shard['snapshot_start']}, {shard['snapshot_end']})"
        )
        dense_dataset = _load_pickle(shard["processed"]["path"])
        if self.use_positions:
            if shard["positions"] is None:
                warn_and_raise(
                    "Training requested separately loaded positions, but no "
                    f"positions shard is available for shard {shard_idx}."
                )
            positions = _load_pickle(shard["positions"]["path"])
            dense_dataset = (*dense_dataset, positions)

        additional_data = None
        if _requires_additional_data(self.additional_data_idx):
            if shard["additional_data"] is None:
                warn_and_raise(
                    "Training requested additional data, but no additional_data "
                    f"shard is available for shard {shard_idx}."
                )
            additional_data = _load_pickle(shard["additional_data"]["path"])

        if self.use_hypergraphs:
            if shard["hypergraphs"] is None:
                warn_and_raise(
                    "Training requested hypergraphs, but no hypergraph shard is "
                    f"available for shard {shard_idx}."
                )
            hypergraphs = _load_pickle(shard["hypergraphs"]["path"])
            dataset = MAPFHypergraphDataset(
                dense_dataset,
                hypergraphs,
                additional_data=additional_data,
                **self.dataset_kwargs,
            )
        else:
            dataset = MAPFGraphDataset(
                dense_dataset,
                additional_data=additional_data,
                **self.dataset_kwargs,
            )

        self._cached_shard_idx = shard_idx
        self._cached_dataset = dataset
        return dataset

    def __getitem__(self, index):
        global_index = self.global_indices[index]
        shard_idx, local_index = self._shard_for_global_index(global_index)
        return self._load_shard_dataset(shard_idx)[local_index]


class ShardedEpisodeSampler(Sampler):
    def __init__(self, snapshot_dataset, *, shuffle=True):
        if not hasattr(snapshot_dataset, "episode_shard_indices"):
            warn_and_raise(
                "Cannot build ShardedEpisodeSampler: snapshot dataset does not "
                "expose episode_shard_indices."
            )
        self.episode_shard_indices = list(snapshot_dataset.episode_shard_indices)
        self.shuffle = shuffle

        self.indices_by_shard = {}
        for sequence_idx, shard_idx in enumerate(self.episode_shard_indices):
            self.indices_by_shard.setdefault(shard_idx, []).append(sequence_idx)

    def __iter__(self):
        shard_ids = list(self.indices_by_shard)
        if self.shuffle:
            random.shuffle(shard_ids)
        for shard_idx in shard_ids:
            indices = list(self.indices_by_shard[shard_idx])
            if self.shuffle:
                random.shuffle(indices)
            yield from indices

    def __len__(self):
        return len(self.episode_shard_indices)


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
