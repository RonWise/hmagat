import argparse
import copy
import pickle
from itertools import compress

import numpy as np
import torch
from loguru import logger

from hmagat.convert_to_imitation_dataset import (
    add_imitation_dataset_args,
    get_imitation_dataset_file_name,
)
from hmagat.dataset_loading import load_dataset
from hmagat.downstream_shards import (
    add_sharded_downstream_args,
    load_stage_manifest,
    validate_stage_manifests,
)
from hmagat.generate_additional_data import (
    add_additional_data_args,
    any_additional_data,
    get_additional_data_file_name,
)
from hmagat.generate_hypergraphs import (
    add_hypergraph_generation_args,
    get_hypergraph_file_name,
)
from hmagat.generate_pos import get_pos_file_name
from hmagat.imitation_dataset_pyg import (
    MAPFGraphDataset,
    MAPFHypergraphDataset,
    validate_sequence_dataset_assumptions,
)
from hmagat.run_expert import add_expert_dataset_args
from hmagat.training_args import add_training_args


def _as_tensor(values):
    if isinstance(values, torch.Tensor):
        return values
    if isinstance(values, np.ndarray):
        return torch.from_numpy(values)
    if len(values) > 0 and isinstance(values[0], torch.Tensor):
        return torch.stack([value.reshape(()) for value in values])
    return torch.from_numpy(np.array(values))


def _split_end_id(unique_map_ids, boundary_index):
    if boundary_index < len(unique_map_ids):
        return unique_map_ids[boundary_index].item()
    return unique_map_ids[-1].item() + 1


def _divide_dataset(dense_dataset, hyper_edge_indices, additional_data, start, end):
    map_ids = _as_tensor(dense_dataset[4])
    mask = torch.logical_and(map_ids >= start, map_ids < end)

    split_hindices = None
    if hyper_edge_indices is not None:
        hindices, hton_indices = hyper_edge_indices
        split_hindices = (
            list(compress(hindices, mask)),
            list(compress(hton_indices, mask)),
        )

    split_additional_data = None
    if additional_data is not None:
        split_additional_data = list(compress(additional_data, mask))

    if isinstance(dense_dataset[0], torch.Tensor):
        split_dense_dataset = tuple(item[mask] for item in dense_dataset)
    else:
        split_dense_dataset = tuple(list(compress(item, mask)) for item in dense_dataset)

    return split_dense_dataset, split_hindices, split_additional_data


def _episode_lengths(graph_map_id):
    values = _as_tensor(graph_map_id).detach().cpu().tolist()
    if not values:
        return []

    lengths = []
    current_id = values[0]
    current_len = 0
    for map_id in values:
        if map_id != current_id:
            lengths.append(current_len)
            current_id = map_id
            current_len = 0
        current_len += 1
    lengths.append(current_len)
    return lengths


def _dense_num_agents(dense_dataset, index):
    node_features = dense_dataset[0][index]
    return int(node_features.shape[0])


def _dense_indices_for_id_range(graph_map_id, start=None, end=None):
    for index, map_id in enumerate(graph_map_id):
        if isinstance(map_id, torch.Tensor):
            map_id = map_id.item()
        if start is not None and map_id < start:
            continue
        if end is not None and map_id >= end:
            continue
        yield index, map_id


def validate_dense_sequence_dataset_assumptions(
    dense_dataset,
    start_id=None,
    end_id=None,
):
    graph_map_id = _as_tensor(dense_dataset[4]).detach().cpu().tolist()
    groups = []
    seen_map_ids = set()
    current_map_id = None
    current_group = []
    expected_num_agents = None

    for snapshot_idx, map_id in _dense_indices_for_id_range(
        graph_map_id,
        start=start_id,
        end=end_id,
    ):
        if current_map_id is None:
            current_map_id = map_id
            current_group = [snapshot_idx]
            seen_map_ids.add(map_id)
            expected_num_agents = _dense_num_agents(dense_dataset, snapshot_idx)
            continue

        if map_id != current_map_id:
            groups.append(current_group)
            if map_id in seen_map_ids:
                raise ValueError(
                    "graph_map_id must be contiguous for sequence grouping; "
                    f"map id {map_id} appears in multiple segments."
                )
            current_map_id = map_id
            current_group = [snapshot_idx]
            seen_map_ids.add(map_id)
            expected_num_agents = _dense_num_agents(dense_dataset, snapshot_idx)
            continue

        current_group.append(snapshot_idx)
        num_agents = _dense_num_agents(dense_dataset, snapshot_idx)
        if num_agents != expected_num_agents:
            raise ValueError(
                "Agent row count changed inside one sequence episode: "
                f"episode map_id {map_id}, snapshot {snapshot_idx}, expected "
                f"{expected_num_agents}, got {num_agents}."
            )

    if current_group:
        groups.append(current_group)

    return {
        "num_episodes": len(groups),
        "num_snapshots": sum(len(group) for group in groups),
        "episode_lengths": [len(group) for group in groups],
        "checked_agent_id": True,
        "agent_id_contract": "row_position",
    }


def _build_snapshot_dataset(args, dense_dataset, hyper_edge_indices, additional_data):
    _, additional_data_idx = any_additional_data(args)
    dataset_kwargs = dict(
        edge_attr_opts=args.edge_attr_opts,
        additional_data_idx=additional_data_idx,
        use_edge_attr=args.use_edge_attr,
        use_edge_attr_for_messages=args.use_edge_attr_for_messages,
    )

    if args.imitation_learning_model == "DirectionalHMAGAT":
        if hyper_edge_indices is None:
            raise ValueError("DirectionalHMAGAT audit requires hypergraph indices.")
        return MAPFHypergraphDataset(
            dense_dataset,
            hyper_edge_indices,
            additional_data=additional_data,
            **dataset_kwargs,
        )

    return MAPFGraphDataset(
        dense_dataset,
        additional_data=additional_data,
        **dataset_kwargs,
    )


def _log_split_summary(name, dataset):
    lengths = _episode_lengths(dataset.graph_map_id)
    if not lengths:
        logger.warning("{} split is empty; skipping sequence assumption checks.", name)
        return False

    logger.info(
        "{} split: snapshots={}, episodes={}, episode_len_min={}, "
        "episode_len_mean={:.2f}, episode_len_max={}",
        name,
        len(dataset),
        len(lengths),
        min(lengths),
        float(np.mean(lengths)),
        max(lengths),
    )
    summary = validate_sequence_dataset_assumptions(dataset)
    logger.info("{} sequence audit summary: {}", name, summary)
    return True


def _log_dense_split_summary(name, dense_dataset, start_id, end_id):
    graph_map_id = _as_tensor(dense_dataset[4])
    mask = torch.logical_and(graph_map_id >= start_id, graph_map_id < end_id)
    split_graph_map_id = graph_map_id[mask]
    lengths = _episode_lengths(split_graph_map_id)
    if not lengths:
        logger.warning("{} split is empty; skipping sequence assumption checks.", name)
        return False

    logger.info(
        "{} split: snapshots={}, episodes={}, episode_len_min={}, "
        "episode_len_mean={:.2f}, episode_len_max={}",
        name,
        int(mask.sum().item()),
        len(lengths),
        min(lengths),
        float(np.mean(lengths)),
        max(lengths),
    )
    summary = validate_dense_sequence_dataset_assumptions(
        dense_dataset,
        start_id=start_id,
        end_id=end_id,
    )
    logger.info("{} sequence audit summary: {}", name, summary)
    return True


def _load_sharded_dense_dataset_entry(entry):
    with open(entry["path"], "rb") as f:
        return pickle.load(f)


def _build_lightweight_graph_dataset_for_audit(dense_dataset):
    dense_dataset = dense_dataset[:5]
    return MAPFGraphDataset(
        dense_dataset,
        additional_data=None,
        additional_data_idx=[None, None, None],
        use_edge_attr=False,
        edge_attr_opts="straight",
        use_edge_attr_for_messages=None,
    )


def _merge_summaries(summaries):
    non_empty = [summary for summary in summaries if summary["num_snapshots"] > 0]
    if not non_empty:
        return {
            "num_episodes": 0,
            "num_snapshots": 0,
            "episode_lengths": [],
            "checked_agent_id": True,
            "agent_id_contract": "row_position",
        }
    return {
        "num_episodes": sum(summary["num_episodes"] for summary in non_empty),
        "num_snapshots": sum(summary["num_snapshots"] for summary in non_empty),
        "episode_lengths": [
            length for summary in non_empty for length in summary["episode_lengths"]
        ],
        "checked_agent_id": all(summary["checked_agent_id"] for summary in non_empty),
        "agent_id_contract": "row_position",
    }


def _required_sharded_manifests(args):
    processed_args = args
    positions_args = args
    if args.load_positions_separately and args.use_edge_attr:
        processed_args = copy.copy(args)
        processed_args.use_edge_attr = False
        positions_args = copy.copy(args)
        positions_args.use_edge_attr = True
        logger.info(
            "Validating sharded audit manifests with processed "
            "use_edge_attr=False and positions use_edge_attr=True because "
            "--load_positions_separately is enabled."
        )

    specs = [("processed_dataset", "processed", processed_args)]
    if args.imitation_learning_model == "DirectionalHMAGAT":
        specs.append(("hypergraphs", "hypergraphs", args))
    load_additional_data, _ = any_additional_data(args)
    if load_additional_data:
        specs.append(("additional_data", "additional_data", args))
    if args.load_positions_separately:
        specs.append(("positions", "positions", positions_args))
    manifests = {
        stage_dir_name: load_stage_manifest(
            stage_args,
            stage_dir_name,
            expected_stage=stage,
        )
        for stage_dir_name, stage, stage_args in specs
    }
    validate_stage_manifests(manifests)
    return manifests


def _log_sharded_dense_split_summary(name, manifest, start_id, end_id):
    summaries = []
    for entry in manifest["entries"]:
        shard_start = entry.get("graph_map_id_start")
        shard_end = entry.get("graph_map_id_end")
        if shard_start is None or shard_end is None:
            logger.warning(
                "Processed shard manifest lacks graph_map_id bounds; loading shard "
                f"for range check: {entry['path']}"
            )
        elif shard_end <= start_id or shard_start >= end_id:
            continue

        dense_dataset = _load_sharded_dense_dataset_entry(entry)
        split_dense_dataset, _, _ = _divide_dataset(
            dense_dataset[:5],
            None,
            None,
            start_id,
            end_id,
        )
        if len(split_dense_dataset[0]) == 0:
            continue
        snapshot_dataset = _build_lightweight_graph_dataset_for_audit(split_dense_dataset)
        summary = validate_sequence_dataset_assumptions(
            snapshot_dataset,
            require_agent_id=True,
        )
        summary["agent_id_contract"] = "row_position"
        summaries.append(summary)

    summary = _merge_summaries(summaries)
    lengths = summary["episode_lengths"]
    if not lengths:
        logger.warning("{} split is empty; skipping sequence assumption checks.", name)
        return False
    logger.info(
        "{} split: snapshots={}, episodes={}, episode_len_min={}, "
        "episode_len_mean={:.2f}, episode_len_max={}",
        name,
        summary["num_snapshots"],
        summary["num_episodes"],
        min(lengths),
        float(np.mean(lengths)),
        max(lengths),
    )
    logger.info("{} sequence audit summary: {}", name, summary)
    return True


def _run_sharded_lightweight_audit(args):
    manifests = _required_sharded_manifests(args)
    manifest = manifests["processed_dataset"]
    num_samples = manifest["total_saved_samples"]
    if num_samples <= 0:
        raise ValueError("Cannot audit an empty sharded processed dataset.")

    first_id = 0
    train_end = int(num_samples * (1 - args.validation_fraction - args.test_fraction))
    validation_end = train_end + int(num_samples * args.validation_fraction)
    train_end = min(train_end, num_samples)
    validation_end = min(validation_end, num_samples)

    logger.info(
        "Sharded dataset ids: first_id={}, num_episode_ids={}, train_end={}, "
        "validation_end={}",
        first_id,
        num_samples,
        train_end,
        validation_end,
    )
    logger.info(
        "Using sharded lightweight sequence audit; required stage manifests are "
        "validated, while hypergraphs, additional data, and positions payloads "
        "are not loaded."
    )
    checked_train = _log_sharded_dense_split_summary(
        "train",
        manifest,
        first_id,
        train_end,
    )
    checked_validation = _log_sharded_dense_split_summary(
        "validation",
        manifest,
        train_end,
        validation_end,
    )
    return checked_train, checked_validation


def main():
    parser = argparse.ArgumentParser(description="Audit sequence dataset assumptions.")
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_additional_data_args(parser)
    parser = add_training_args(parser)
    parser = add_sharded_downstream_args(parser)
    parser.add_argument(
        "--materialize_audit_dataset",
        action="store_true",
        default=False,
        help=(
            "Build PyG snapshot datasets during audit. The default lightweight audit "
            "checks dense sequence metadata without loading hypergraphs, additional "
            "data, or positions."
        ),
    )
    args = parser.parse_args()

    if args.use_shards:
        if args.materialize_audit_dataset:
            message = (
                "Materialized sequence audit is not implemented for sharded datasets. "
                "Run sharded lightweight audit without --materialize_audit_dataset, "
                "or use legacy single-file artifacts for materialized audit."
            )
            logger.warning(message)
            raise ValueError(message)
        checked_train, checked_validation = _run_sharded_lightweight_audit(args)
        if checked_train or checked_validation:
            logger.warning(
                "Sequence audit cannot prove physical temporal order without an "
                "explicit timestep field; it verifies contiguous graph_map_id "
                "grouping, first_step consistency, agent count consistency, and "
                "stable agent_id when present."
            )
        return

    logger.info("Loading processed dataset for sequence audit.")
    dense_dataset = load_dataset(
        [get_imitation_dataset_file_name],
        "processed_dataset",
        args,
    )

    graph_map_id = _as_tensor(dense_dataset[4])
    unique_map_ids = torch.unique(graph_map_id, sorted=True)
    if len(unique_map_ids) == 0:
        raise ValueError("Cannot audit an empty dataset.")

    num_samples = len(unique_map_ids)
    train_boundary = int(
        num_samples * (1 - args.validation_fraction - args.test_fraction)
    )
    validation_boundary = train_boundary + int(num_samples * args.validation_fraction)
    train_boundary = min(train_boundary, num_samples)
    validation_boundary = min(validation_boundary, num_samples)

    first_id = unique_map_ids[0].item()
    train_end = _split_end_id(unique_map_ids, train_boundary)
    validation_end = _split_end_id(unique_map_ids, validation_boundary)

    logger.info(
        "Dataset ids: first_id={}, num_episode_ids={}, train_end={}, "
        "validation_end={}",
        first_id,
        num_samples,
        train_end,
        validation_end,
    )

    if args.materialize_audit_dataset:
        logger.warning(
            "Materialized sequence audit loads hypergraphs/additional data/positions "
            "and may require much more RAM than the default lightweight audit."
        )
        if args.load_positions_separately:
            logger.info("Loading separately stored agent positions.")
            agent_pos = load_dataset([get_pos_file_name], "positions", args)
            dense_dataset = (*dense_dataset, agent_pos)

        hyper_edge_indices = None
        if args.imitation_learning_model == "DirectionalHMAGAT":
            logger.info("Loading hypergraph indices for DirectionalHMAGAT audit.")
            hyper_edge_indices = load_dataset(
                [get_hypergraph_file_name],
                "hypergraphs",
                args,
            )

        additional_data = None
        load_additional_data, _ = any_additional_data(args)
        if load_additional_data:
            logger.info("Loading additional data for sequence audit.")
            additional_data = load_dataset(
                [get_additional_data_file_name],
                "additional_data",
                args,
            )

        train_parts = _divide_dataset(
            dense_dataset,
            hyper_edge_indices,
            additional_data,
            first_id,
            train_end,
        )
        validation_parts = _divide_dataset(
            dense_dataset,
            hyper_edge_indices,
            additional_data,
            train_end,
            validation_end,
        )

        train_dataset = _build_snapshot_dataset(args, *train_parts)
        validation_dataset = _build_snapshot_dataset(args, *validation_parts)

        checked_train = _log_split_summary("train", train_dataset)
        checked_validation = _log_split_summary("validation", validation_dataset)
    else:
        logger.info(
            "Using lightweight sequence audit; hypergraphs, additional data, and "
            "positions are not loaded."
        )
        checked_train = _log_dense_split_summary(
            "train",
            dense_dataset,
            first_id,
            train_end,
        )
        checked_validation = _log_dense_split_summary(
            "validation",
            dense_dataset,
            train_end,
            validation_end,
        )

    if checked_train or checked_validation:
        logger.warning(
            "Sequence audit cannot prove physical temporal order without an explicit "
            "timestep field; it verifies contiguous graph_map_id grouping, first_step "
            "consistency, agent count consistency, and stable agent_id when present."
        )


if __name__ == "__main__":
    main()
