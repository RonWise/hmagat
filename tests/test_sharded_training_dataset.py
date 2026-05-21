import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from loguru import logger

from hmagat.downstream_shards import write_stage_manifest, write_stage_shard
from hmagat.imitation_dataset_pyg import (
    MAPFGraphDataset,
    MAPFHypergraphDataset,
    MAPFSequenceDataset,
    validate_sequence_dataset_assumptions,
)
from hmagat.sharded_training_dataset import ShardedEpisodeBatchSampler
from hmagat.sharded_training_dataset import ShardedSnapshotDataset
from hmagat.sharded_training_dataset import ShardedEpisodeSampler
from hmagat.sharded_training_dataset import ShardedSequenceEpisode
from hmagat.sharded_training_dataset import ShardedSequenceDataset
from hmagat.sharded_training_dataset import build_sharded_snapshot_datasets


def _args(tmp_dir, name="paper", num_samples=4):
    return SimpleNamespace(
        dataset_dir=tmp_dir,
        override_name=name,
        num_samples=num_samples,
        dataset_seed=42,
        sharded_dataset_cache_size=1,
        obs_radius=1,
        save_termination_state=True,
        expert_algorithm="LaCAM",
        obstacle_density_max=0.7,
        ensure_grid_config_is_generatable=True,
        comm_radius=7,
        num_neighbour_cutoff=None,
        neighbour_cutoff_method="closest",
        use_lists=True,
        distance_metric="euclidean",
        random_edge_probs=None,
        use_edge_attr=False,
        load_positions_separately=False,
        hypergraph_comm_radius=7,
        take_all_seeds=False,
        hypergraph_max_group_size=None,
        hyperedge_generation_method="kmeans",
        add_hypergraph_self_loop=False,
        comm_self=True,
        hypergraph_max_neighbours=None,
        hypergraph_max_dist_threshold=3,
        hypergraph_max_dist_frac=0.8,
        hypergraph_max_clique_size=4,
        hypergraph_initial_colperc=0.1,
        hypergraph_final_colperc=0.1,
        hypergraph_wait_one=True,
        hypergraph_time_period=1,
        hypergraph_num_updates=10,
        add_data_num_previous_actions=None,
        add_data_cost_to_go=True,
        normalize_cost_to_go=True,
        clamp_cost_to_go=1.0,
        add_data_greedy_action=False,
        clamped_values_doubled=False,
        validation_fraction=0.0,
        test_fraction=0.0,
        skip_validation=True,
    )


def _dense_shard(start_map_id, episode_lengths, num_agents=3):
    node_features = []
    adj = []
    actions = []
    terminated = []
    graph_map_id = []
    for episode_offset, length in enumerate(episode_lengths):
        map_id = start_map_id + episode_offset
        for timestep in range(length):
            value = float(10 * map_id + timestep)
            node_features.append(torch.full((num_agents, 3, 5, 5), value))
            adj.append(torch.eye(num_agents))
            actions.append(torch.full((num_agents,), timestep % 5, dtype=torch.long))
            terminated.append(torch.zeros(num_agents, dtype=torch.bool))
            graph_map_id.append(map_id)
    return (
        node_features,
        adj,
        actions,
        terminated,
        torch.tensor(graph_map_id, dtype=torch.long),
    )


def _positions_for_dense(dense_dataset):
    positions = []
    for idx, node_features in enumerate(dense_dataset[0]):
        num_agents = node_features.shape[0]
        positions.append(
            torch.stack(
                [
                    torch.arange(num_agents, dtype=torch.long),
                    torch.full((num_agents,), idx, dtype=torch.long),
                ],
                dim=1,
            )
        )
    return positions


def _hypergraphs_for_dense(dense_dataset):
    ntoh = []
    hton = []
    for node_features in dense_dataset[0]:
        num_agents = node_features.shape[0]
        ntoh.append(
            torch.tensor(
                [
                    list(range(num_agents)),
                    [agent_idx // 2 for agent_idx in range(num_agents)],
                ],
                dtype=torch.long,
            )
        )
        hton.append(
            torch.tensor(
                [
                    [agent_idx // 2 for agent_idx in range(num_agents)],
                    list(range(num_agents)),
                ],
                dtype=torch.long,
            )
        )
    return ntoh, hton


def _additional_for_dense(dense_dataset):
    result = []
    for idx, node_features in enumerate(dense_dataset[0]):
        result.append(
            [
                torch.full(
                    (
                        node_features.shape[0],
                        node_features.shape[2],
                        node_features.shape[3],
                    ),
                    float(idx),
                )
            ]
        )
    return result


def _merge_sequence_payloads(payloads):
    merged = []
    for field_idx in range(len(payloads[0])):
        if field_idx == 4:
            merged.append(torch.cat([payload[field_idx] for payload in payloads]))
        else:
            field = []
            for payload in payloads:
                field.extend(payload[field_idx])
            merged.append(field)
    return tuple(merged)


def _write_stage_payloads(tmp_dir, args, stage_dir_name, stage, payloads):
    entries = []
    sample_start = 0
    for shard_idx, payload in enumerate(payloads):
        graph_map_ids = payload[4] if stage == "processed" else None
        saved_samples = 2
        sample_end = sample_start + saved_samples
        kwargs = {}
        if graph_map_ids is not None:
            kwargs["graph_map_id_start"] = int(graph_map_ids[0].item())
            kwargs["graph_map_id_end"] = int(graph_map_ids[-1].item()) + 1
        entries.append(
            write_stage_shard(
                args,
                stage_dir_name=stage_dir_name,
                stage=stage,
                shard_idx=shard_idx,
                sample_start=sample_start,
                sample_end=sample_end,
                payload=payload,
                saved_samples=saved_samples,
                snapshot_count=(
                    len(payload[0])
                    if stage in {"processed", "hypergraphs"}
                    else len(payload)
                ),
                **kwargs,
            )
        )
        sample_start = sample_end
    return write_stage_manifest(args, stage_dir_name, stage, entries)


def _capture_warnings(func):
    records = []
    sink_id = logger.add(
        lambda message: records.append(message.record),
        level="WARNING",
        format="{message}",
    )
    try:
        func()
    finally:
        logger.remove(sink_id)
    return [record["message"] for record in records]


class ShardedTrainingDatasetTest(unittest.TestCase):
    def test_sharded_graph_dataset_matches_legacy_graph_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            merged = _merge_sequence_payloads([shard_0, shard_1])
            legacy = MAPFGraphDataset(merged, use_edge_attr=False)
            sharded = ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )

            self.assertEqual(len(sharded), len(legacy))
            self.assertEqual(sharded.graph_map_id, merged[4].tolist())
            for idx in range(len(legacy)):
                expected = legacy[idx]
                actual = sharded[idx]
                torch.testing.assert_close(actual.x, expected.x)
                torch.testing.assert_close(actual.edge_index, expected.edge_index)
                torch.testing.assert_close(actual.edge_weight, expected.edge_weight)
                torch.testing.assert_close(actual.y, expected.y)
                self.assertEqual(actual.first_step.item(), expected.first_step.item())

    def test_sharded_hypergraph_dataset_matches_legacy_with_positions_and_additional_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            args.use_edge_attr = True
            args.load_positions_separately = True
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            processed_args = SimpleNamespace(**vars(args))
            processed_args.use_edge_attr = False
            _write_stage_payloads(
                tmp_dir,
                processed_args,
                "processed_dataset",
                "processed",
                [shard_0, shard_1],
            )
            hyper_payloads = [_hypergraphs_for_dense(shard_0), _hypergraphs_for_dense(shard_1)]
            additional_payloads = [_additional_for_dense(shard_0), _additional_for_dense(shard_1)]
            position_payloads = [_positions_for_dense(shard_0), _positions_for_dense(shard_1)]
            _write_stage_payloads(tmp_dir, args, "hypergraphs", "hypergraphs", hyper_payloads)
            _write_stage_payloads(
                tmp_dir, args, "additional_data", "additional_data", additional_payloads
            )
            _write_stage_payloads(tmp_dir, args, "positions", "positions", position_payloads)

            merged_processed = _merge_sequence_payloads([shard_0, shard_1])
            merged_positions = position_payloads[0] + position_payloads[1]
            merged_hypergraphs = (
                hyper_payloads[0][0] + hyper_payloads[1][0],
                hyper_payloads[0][1] + hyper_payloads[1][1],
            )
            merged_additional = additional_payloads[0] + additional_payloads[1]
            legacy = MAPFHypergraphDataset(
                (*merged_processed, merged_positions),
                merged_hypergraphs,
                use_edge_attr=True,
                additional_data=merged_additional,
                additional_data_idx=[0, None, None],
                use_edge_attr_for_messages="positions+manhattan",
            )
            sharded = ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=True,
                use_edge_attr=True,
                additional_data_idx=[0, None, None],
                use_edge_attr_for_messages="positions+manhattan",
            )

            for idx in range(len(legacy)):
                expected = legacy[idx]
                actual = sharded[idx]
                torch.testing.assert_close(actual.x, expected.x)
                torch.testing.assert_close(actual.edge_index_src, expected.edge_index_src)
                torch.testing.assert_close(actual.edge_index_dst, expected.edge_index_dst)
                torch.testing.assert_close(
                    actual.hton_edge_index_src, expected.hton_edge_index_src
                )
                torch.testing.assert_close(
                    actual.hton_edge_index_dst, expected.hton_edge_index_dst
                )
                torch.testing.assert_close(actual.edge_attr, expected.edge_attr)

            stats = sharded.runtime_stats_snapshot()
            self.assertEqual(stats["shard_loads"], 2)
            self.assertGreaterEqual(stats["processed_payload_load_sec"], 0.0)
            self.assertGreaterEqual(stats["positions_payload_load_sec"], 0.0)
            self.assertGreaterEqual(stats["additional_data_payload_load_sec"], 0.0)
            self.assertGreaterEqual(stats["hypergraph_payload_load_sec"], 0.0)
            self.assertGreaterEqual(stats["shard_dataset_materialization_sec"], 0.0)

    def test_sharded_sequence_dataset_preserves_episode_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            sharded = ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )
            sequences = MAPFSequenceDataset(sharded)

            self.assertEqual(len(sequences), 4)
            self.assertEqual([len(sequences[idx]) for idx in range(4)], [2, 1, 1, 2])
            self.assertTrue(sequences[0][0].first_step.item())
            self.assertFalse(sequences[0][1].first_step.item())

    def test_native_sharded_sequence_dataset_matches_snapshot_wrapped_sequences(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            sharded_snapshot = ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )
            wrapped_sequence = MAPFSequenceDataset(sharded_snapshot)
            native_sequence = ShardedSequenceDataset.from_snapshot_dataset(
                sharded_snapshot
            )

            self.assertEqual(len(native_sequence), len(wrapped_sequence))
            self.assertEqual(native_sequence.episode_shard_indices, [0, 0, 1, 1])
            for episode_idx in range(len(native_sequence)):
                self.assertIsInstance(
                    native_sequence[episode_idx], ShardedSequenceEpisode
                )
                self.assertEqual(
                    len(native_sequence[episode_idx]),
                    len(wrapped_sequence[episode_idx]),
                )
                for actual, expected in zip(
                    native_sequence[episode_idx], wrapped_sequence[episode_idx]
                ):
                    torch.testing.assert_close(actual.x, expected.x)
                    torch.testing.assert_close(actual.y, expected.y)
                    torch.testing.assert_close(
                        actual.terminated, expected.terminated
                    )
                    self.assertEqual(
                        bool(actual.first_step.item()),
                        bool(expected.first_step.item()),
                    )

    def test_native_sharded_sequence_episode_defers_shard_access_until_iteration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            native_sequence = ShardedSequenceDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )

            episode = native_sequence[0]
            stats_before_access = native_sequence.runtime_stats_snapshot()

            self.assertEqual(stats_before_access["shard_accesses"], 0)
            self.assertEqual(stats_before_access["shard_loads"], 0)

            first_timestep = episode[0]
            stats_after_access = native_sequence.runtime_stats_snapshot()

            self.assertTrue(first_timestep.first_step.item())
            self.assertEqual(stats_after_access["shard_accesses"], 1)
            self.assertEqual(stats_after_access["shard_loads"], 1)

    def test_native_sharded_sequence_runtime_trace_records_caller_and_context(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            native_sequence = ShardedSequenceDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )
            native_sequence.set_runtime_context(
                dataset_role="train_sequence",
                phase="train",
                epoch=2,
                batch_idx=7,
            )

            timestep_items = [[], []]
            timestep_active_indices = [[], []]
            native_sequence[0]._append_timestep_items(
                timestep_items, timestep_active_indices, 0
            )
            stats = native_sequence.runtime_stats_snapshot()

            self.assertEqual(stats["dataset_role"], "train_sequence")
            self.assertEqual(stats["phase"], "train")
            self.assertEqual(stats["epoch"], 2)
            self.assertEqual(stats["batch_idx"], 7)
            self.assertGreaterEqual(stats["load_events_recorded"], 1)
            recent_events = stats["recent_shard_events"]
            self.assertTrue(recent_events)
            self.assertIn("cache_miss_load", recent_events[0])
            self.assertIn("caller=sequence_episode_append_timestep_items", recent_events[0])
            self.assertIn("dataset_role=train_sequence", recent_events[0])
            self.assertIn("phase=train", recent_events[0])
            self.assertIn("epoch=2", recent_events[0])
            self.assertIn("batch_idx=7", recent_events[0])

    def test_sharded_episode_sampler_groups_sequences_by_shard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            sharded = ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )
            sampler = ShardedEpisodeSampler(sharded, shuffle=False)

            self.assertEqual(sharded.episode_shard_indices, [0, 0, 1, 1])
            self.assertEqual(list(sampler), [0, 1, 2, 3])

    def test_sharded_episode_batch_sampler_keeps_batches_within_shard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            sharded_snapshot = ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )
            native_sequence = ShardedSequenceDataset.from_snapshot_dataset(
                sharded_snapshot
            )
            batch_sampler = ShardedEpisodeBatchSampler(
                native_sequence,
                batch_size=2,
                shuffle=False,
            )

            self.assertEqual(list(batch_sampler), [[0, 1], [2, 3]])

    def test_sharded_snapshot_dataset_lru_cache_reuses_loaded_shards(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            args.sharded_dataset_cache_size = 2
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            sharded = ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )

            _ = sharded[0]
            _ = sharded[3]
            _ = sharded[0]
            stats = sharded.runtime_stats_snapshot()

            self.assertEqual(stats["shard_loads"], 2)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["cache_evictions"], 0)
            self.assertEqual(stats["cache_size"], 2)

    def test_sharded_snapshot_dataset_lru_cache_evicts_when_capacity_is_one(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            args.sharded_dataset_cache_size = 1
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            sharded = ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )

            _ = sharded[0]
            _ = sharded[3]
            _ = sharded[0]
            stats = sharded.runtime_stats_snapshot()

            self.assertEqual(stats["shard_loads"], 3)
            self.assertEqual(stats["cache_hits"], 0)
            self.assertEqual(stats["cache_evictions"], 2)
            self.assertEqual(stats["cache_size"], 1)
            recent_events = stats["recent_shard_events"]
            self.assertTrue(
                any("cache_evict" in event and "shard=0" in event for event in recent_events)
            )
            self.assertTrue(
                any(
                    "cache_miss_load" in event and "caller=snapshot_getitem" in event
                    for event in recent_events
                )
            )

    def test_sharded_training_index_sidecar_is_written_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            index_path = os.path.join(
                tmp_dir, "processed_dataset", "shards", "training_index.json"
            )
            self.assertFalse(os.path.exists(index_path))

            ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )
            self.assertTrue(os.path.exists(index_path))

            with mock.patch(
                "hmagat.sharded_training_dataset._load_pickle",
                side_effect=AssertionError("processed shards should not be read"),
            ):
                sharded = ShardedSnapshotDataset(
                    args,
                    episode_start=0,
                    episode_end=4,
                    use_hypergraphs=False,
                    use_edge_attr=False,
                    additional_data_idx=[None, None, None],
                )

            self.assertEqual(len(sharded), 6)
            self.assertEqual(sharded.graph_map_id, [0, 0, 1, 2, 3, 3])

    def test_sharded_training_index_infers_original_sample_ids_from_graph_map_range_without_raw_shards(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            manifest_path = os.path.join(
                tmp_dir, "processed_dataset", "shards", "manifest.json"
            )
            with open(manifest_path) as f:
                import json

                manifest = json.load(f)
            for entry in manifest["entries"]:
                entry.pop("original_sample_ids", None)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            warnings = _capture_warnings(
                lambda: ShardedSnapshotDataset(
                    args,
                    episode_start=0,
                    episode_end=4,
                    use_hypergraphs=False,
                    use_edge_attr=False,
                    additional_data_idx=[None, None, None],
                )
            )

            self.assertTrue(
                any(
                    "inferring them from contiguous graph_map_id range"
                    in message
                    for message in warnings
                )
            )

    def test_sharded_training_split_builds_shared_index_once(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            args.skip_validation = False
            args.validation_fraction = 0.5
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            import hmagat.sharded_training_dataset as sharded_training_dataset

            original_load_pickle = sharded_training_dataset._load_pickle
            processed_loads = []

            def counting_load_pickle(path):
                if "processed_dataset" in str(path):
                    processed_loads.append(str(path))
                return original_load_pickle(path)

            with mock.patch(
                "hmagat.sharded_training_dataset._load_pickle",
                side_effect=counting_load_pickle,
            ):
                train_dataset, validation_dataset, train_end, validation_end = (
                    build_sharded_snapshot_datasets(
                        args,
                        hypergraph_model=False,
                        additional_data_idx=[None, None, None],
                        dataset_kwargs={
                            "use_edge_attr": False,
                            "additional_data_idx": [None, None, None],
                        },
                    )
                )

            self.assertEqual(train_end, 2)
            self.assertEqual(validation_end, 4)
            self.assertEqual(len(train_dataset), 3)
            self.assertEqual(len(validation_dataset), 3)
            self.assertEqual(len(processed_loads), 2)

    def test_sharded_training_dataset_rejects_stage_manifest_mismatch_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            hyper_payloads = [_hypergraphs_for_dense(shard_0), _hypergraphs_for_dense(shard_1)]
            manifest = _write_stage_payloads(
                tmp_dir, args, "hypergraphs", "hypergraphs", hyper_payloads
            )
            manifest["entries"][1]["snapshot_count"] += 1
            manifest_path = os.path.join(tmp_dir, "hypergraphs", "shards", "manifest.json")
            with open(manifest_path, "w") as f:
                import json

                json.dump(manifest, f)

            def build_dataset():
                with self.assertRaises(ValueError):
                    ShardedSnapshotDataset(
                        args,
                        episode_start=0,
                        episode_end=4,
                        use_hypergraphs=True,
                        use_edge_attr=False,
                        additional_data_idx=[None, None, None],
                    )

            warnings = _capture_warnings(build_dataset)

            self.assertTrue(any("manifest entry mismatch" in msg for msg in warnings))

    def test_sharded_training_split_allows_empty_validation_when_validation_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )

            train_dataset, validation_dataset, train_end, validation_end = (
                build_sharded_snapshot_datasets(
                    args,
                    hypergraph_model=False,
                    additional_data_idx=[None, None, None],
                    dataset_kwargs={
                        "use_edge_attr": False,
                        "additional_data_idx": [None, None, None],
                    },
                )
            )

            self.assertEqual(train_end, 4)
            self.assertEqual(validation_end, 4)
            self.assertEqual(len(train_dataset), 6)
            self.assertEqual(len(validation_dataset), 0)

    def test_sharded_training_split_rejects_empty_validation_when_validation_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            args.skip_validation = False
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )

            def build_dataset():
                with self.assertRaises(ValueError):
                    build_sharded_snapshot_datasets(
                        args,
                        hypergraph_model=False,
                        additional_data_idx=[None, None, None],
                        dataset_kwargs={
                            "use_edge_attr": False,
                            "additional_data_idx": [None, None, None],
                        },
                    )

            warnings = _capture_warnings(build_dataset)

            self.assertTrue(any("split is empty" in msg for msg in warnings))

    def test_sequence_validation_emits_progress_log_when_label_is_provided(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(tmp_dir)
            shard_0 = _dense_shard(0, [2, 1])
            shard_1 = _dense_shard(2, [1, 2])
            _write_stage_payloads(
                tmp_dir, args, "processed_dataset", "processed", [shard_0, shard_1]
            )
            sharded = ShardedSnapshotDataset(
                args,
                episode_start=0,
                episode_end=4,
                use_hypergraphs=False,
                use_edge_attr=False,
                additional_data_idx=[None, None, None],
            )
            records = []
            sink_id = logger.add(
                lambda message: records.append(message.record),
                level="INFO",
                format="{message}",
            )
            try:
                validate_sequence_dataset_assumptions(
                    sharded,
                    progress_label="sequence preflight smoke",
                )
            finally:
                logger.remove(sink_id)
            info_messages = [record["message"] for record in records]

            self.assertTrue(
                any("sequence preflight smoke" in message for message in info_messages)
            )


if __name__ == "__main__":
    unittest.main()
