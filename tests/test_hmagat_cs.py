import argparse
import csv
import os
import pickle
import pathlib
import random
import tempfile
import unittest
import weakref
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch
from loguru import logger
from pogema import GridConfig, pogema_v0
from tensorboard.backend.event_processing import event_accumulator
from torch_geometric.data import Batch, Data
import test_imitation_learning_pyg

from hmagat import audit_sequence_dataset
from hmagat import convert_to_imitation_dataset
from hmagat import generate_additional_data
from hmagat import train_imitation_learning_pyg
from hmagat.audit_sequence_dataset import validate_dense_sequence_dataset_assumptions
from hmagat.convert_to_imitation_dataset import _as_torch_tensor
from hmagat.checkpointing import (
    TRAINING_CHECKPOINT_TYPE,
    TRAINING_CHECKPOINT_VERSION,
    extract_model_state_dict_from_checkpoint,
    load_partial_checkpoint_into_model,
    resolve_evaluation_checkpoint_path,
)
from hmagat.dataset_loading import load_dataset
from hmagat.imitation_dataset_pyg import (
    MAPFGraphDataset,
    MAPFHypergraphDataset,
    MAPFSequenceBatch,
    MAPFSequenceDataset,
    SequenceBatchCollator,
    collate_mapf_sequences,
    group_indices_by_graph_map_id,
    validate_sequence_dataset_assumptions,
)
from hmagat.modules.agents import (
    DecentralPlannerGATNet,
    get_model,
    load_partial_state_dict,
    validate_partial_load_compatibility,
)
from hmagat.runtime_data_generation import get_runtime_data_generator
from hmagat.sequence_training import (
    SequenceStepRuntimeTracker,
    compute_sequence_loss,
)
from hmagat.training_args import add_training_args, validate_training_args_contract
from hmagat.lr_scheduler import get_estimated_total_number_of_steps
from hmagat.sharded_training_dataset import (
    ShardedEpisodeSampler,
    _build_training_index_payload,
    load_sequence_audit_stamp,
    sequence_audit_stamp_path,
    write_sequence_audit_stamp,
)
from hmagat.train_imitation_learning_pyg import (
    EpochRuntimeTracker,
    _apply_cs_warmup_freeze_baseline,
    _apply_gradient_clipping,
    _build_dataloader_kwargs,
    _build_sharded_dataset_state,
    _build_training_checkpoint_payload,
    _default_batch_metrics_csv_path,
    _default_validation_metrics_csv_path,
    _create_tensorboard_writer,
    _create_optimizer,
    _effective_num_batches,
    _log_tensorboard_scalars,
    _load_resume_training_checkpoint,
    _restore_rng_state,
    _restore_resume_training_state,
    _sharded_sequence_loader_risk_warnings,
    _validate_checkpoint_source_args,
    _validate_resume_checkpoint_args,
    _validate_resume_dataset_state,
    _validate_amp_args,
    _validation_rollout_graph_ids,
    _validate_cs_warmup_freeze_baseline,
    _validate_and_warn_batch_limits,
    _resolve_sequence_dataset_validation_mode,
    _sequence_runtime_metrics,
)
from hmagat.downstream_shards import write_stage_manifest, write_stage_shard


def _graph_data(num_agents=4):
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    return Data(edge_index=edge_index, num_nodes=num_agents)


def _hypergraph_data():
    return Data(
        edge_index_src=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        edge_index_dst=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        hton_edge_index_src=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        hton_edge_index_dst=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        num_nodes=4,
    )


def _observations(num_agents=4, fov=5):
    image_size = (fov + 1) * 2 + 1
    return torch.randn(num_agents, 3, image_size, image_size)


def _tiny_dense_dataset(num_maps=2, num_agents=4, fov=5):
    node_features = []
    adjacency = []
    target_actions = []
    terminated = []
    graph_map_id = []
    for map_id in range(num_maps):
        node_features.append(_observations(num_agents=num_agents, fov=fov))
        adj = torch.ones(num_agents, num_agents, dtype=torch.float32)
        adj.fill_diagonal_(0.0)
        adjacency.append(adj)
        target_actions.append(
            torch.tensor(
                [(agent_id + map_id) % 5 for agent_id in range(num_agents)],
                dtype=torch.long,
            )
        )
        terminated.append(torch.zeros(num_agents, dtype=torch.bool))
        graph_map_id.append(map_id)
    return (
        torch.stack(node_features),
        torch.stack(adjacency),
        torch.stack(target_actions),
        torch.stack(terminated),
        torch.tensor(graph_map_id, dtype=torch.long),
    )


def _write_tiny_processed_dataset(tmp_dir, override_name, *, fov=5, num_maps=2):
    processed_dir = os.path.join(tmp_dir, "processed_dataset")
    os.makedirs(processed_dir, exist_ok=True)
    args = SimpleNamespace(
        override_name=override_name,
        comm_radius=7,
        distance_metric="euclidean",
        random_edge_probs=None,
        num_neighbour_cutoff=None,
        neighbour_cutoff_method="closest",
        load_positions_separately=False,
        use_edge_attr=False,
    )
    dataset_path = os.path.join(
        processed_dir,
        convert_to_imitation_dataset.get_imitation_dataset_file_name(args),
    )
    with open(dataset_path, "wb") as f:
        pickle.dump(_tiny_dense_dataset(num_maps=num_maps, fov=fov), f)
    return dataset_path


def _write_tiny_processed_shard_manifest(tmp_dir, override_name, *, num_samples=4):
    shard_dir = os.path.join(tmp_dir, "processed_dataset", "shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f"{override_name}_processed_shard_000.pkl")
    with open(shard_path, "wb") as f:
        pickle.dump(("unused",), f)

    args = SimpleNamespace(
        dataset_dir=tmp_dir,
        override_name=override_name,
        num_samples=num_samples,
        dataset_seed=42,
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
        validation_fraction=0.25,
        test_fraction=0.25,
    )
    entries = [
        {
            "file_name": os.path.basename(shard_path),
            "path": shard_path,
            "shard_idx": 0,
            "sample_start": 0,
            "sample_end": num_samples,
            "saved_samples": 3,
            "snapshot_count": 6,
            "graph_map_id_start": 0,
            "graph_map_id_end": 3,
            "original_sample_ids": [0, 2, 3],
        }
    ]
    manifest = write_stage_manifest(args, "processed_dataset", "processed", entries)
    return args, manifest


def _write_tiny_sharded_processed_dataset(tmp_dir, override_name):
    args = SimpleNamespace(
        dataset_dir=tmp_dir,
        override_name=override_name,
        num_samples=2,
        dataset_seed=42,
        obs_radius=5,
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
        validation_fraction=0.0,
        test_fraction=0.0,
    )
    dense_dataset = (
        [_observations() for _ in range(3)],
        [torch.ones(4, 4) for _ in range(3)],
        [torch.zeros(4, dtype=torch.long) for _ in range(3)],
        [torch.zeros(4, dtype=torch.bool) for _ in range(3)],
        torch.tensor([0, 0, 1], dtype=torch.long),
    )
    entry = write_stage_shard(
        args,
        stage_dir_name="processed_dataset",
        stage="processed",
        shard_idx=0,
        sample_start=0,
        sample_end=2,
        payload=dense_dataset,
        saved_samples=2,
        snapshot_count=3,
        graph_map_id_start=0,
        graph_map_id_end=2,
        original_sample_ids=[0, 1],
    )
    processed_manifest = write_stage_manifest(
        args, "processed_dataset", "processed", [entry]
    )
    write_sequence_audit_stamp(
        args,
        processed_manifest,
        checked_train=True,
        checked_validation=False,
    )
    return args


def _graph_step(obs, first_step):
    data = _graph_data(num_agents=obs.shape[0])
    data.x = obs
    data.y = torch.zeros(obs.shape[0], dtype=torch.long)
    data.terminated = torch.zeros(obs.shape[0], dtype=torch.bool)
    data.first_step = torch.tensor(first_step)
    return data


class _SnapshotList:
    def __init__(self, snapshots, graph_map_id):
        self.snapshots = snapshots
        self.graph_map_id = graph_map_id

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, index):
        return self.snapshots[index]


class _LightweightSequence:
    def __init__(self, timesteps):
        self.timesteps = list(timesteps)

    def __len__(self):
        return len(self.timesteps)

    def __getitem__(self, index):
        return self.timesteps[index]

    def _append_timestep_items(
        self, timestep_items, timestep_active_indices, episode_idx
    ):
        for timestep, data in enumerate(self.timesteps):
            timestep_items[timestep].append(data)
            timestep_active_indices[timestep].append(episode_idx)


def _reference_collate_mapf_sequences(sequences):
    if len(sequences) == 0:
        raise ValueError("Cannot collate an empty sequence batch.")

    sequence_lengths = torch.tensor([len(sequence) for sequence in sequences])
    max_length = int(torch.max(sequence_lengths).item())
    if max_length == 0:
        raise ValueError("Cannot collate sequence batch with no timesteps.")

    timesteps = []
    active_episode_indices = []
    for timestep in range(max_length):
        timestep_items = []
        active_indices = []
        for episode_idx, sequence in enumerate(sequences):
            if timestep >= len(sequence):
                continue
            timestep_items.append(sequence[timestep])
            active_indices.append(episode_idx)
        batch = Batch.from_data_list(timestep_items)
        timesteps.append(batch)
        active_episode_indices.append(torch.tensor(active_indices, dtype=torch.long))

    return MAPFSequenceBatch(
        timesteps=timesteps,
        active_episode_indices=active_episode_indices,
        sequence_lengths=sequence_lengths,
    )


def _model(**overrides):
    kwargs = dict(
        FOV=5,
        numInputFeatures=8,
        num_attention_heads=1,
        num_gnn_layers=1,
        concat_attention=True,
        cnn_mode="basic-CNN",
        embedding_sizes_gnn=[8],
        num_classes=5,
        use_dropout=False,
        gnn_type="MAGAT",
        gnn_kwargs={},
    )
    kwargs.update(overrides)
    model = DecentralPlannerGATNet(**kwargs)
    model.reset_parameters()
    model.eval()
    return model


def _checkpoint_compatible_args(**overrides):
    kwargs = dict(
        imitation_learning_model="DirectionalHMAGAT",
        obs_radius=5,
        embedding_size=128,
        num_attention_heads=1,
        cnn_mode="ResNetLarge_withMLP",
        use_edge_weights=False,
        use_edge_attr=True,
        edge_dim=None,
        model_residuals=None,
        hyperedge_feature_generator="magat",
        use_edge_attr_for_messages="positions+manhattan",
        edge_attr_cnn_mode="MLP",
        final_feature_generator="magat",
        coordination_state_size=32,
        coordination_state_update="gru",
        num_gnn_layers=3,
        pre_gnn_embedding_size=None,
        pre_gnn_num_mlp_layers=None,
        module_residual=None,
        lin_x_before_additional_data=False,
        add_data_cost_to_go=True,
        add_data_greedy_action=False,
        add_data_num_previous_actions=None,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _checkpoint_runtime_args(**overrides):
    kwargs = vars(_checkpoint_compatible_args(model_residuals="all")).copy()
    kwargs.update(
        dict(
            edge_attr_opts="straight",
            comm_radius=7,
            num_neighbour_cutoff=None,
            neighbour_cutoff_method="closest",
            distance_metric="euclidean",
            random_edge_probs=None,
            hypergraph_comm_radius=7,
            hypergraph_max_group_size=None,
            hyperedge_generation_method="kmeans",
            comm_self=True,
            hypergraph_max_neighbours=None,
            hypergraph_max_dist_threshold=3,
            hypergraph_max_dist_frac=0.8,
            hypergraph_max_clique_size=4,
            hypergraph_initial_colperc=0.1,
            hypergraph_final_colperc=0.1,
            hypergraph_wait_one=True,
            add_hypergraph_self_loop=False,
            hypergraph_num_updates=10,
            hypergraph_time_period=1,
            normalize_cost_to_go=True,
            clamp_cost_to_go=1.0,
            clamped_values_doubled=False,
        )
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _capture_loguru_warnings(func):
    records = []
    sink_id = logger.add(
        lambda message: records.append(message.record),
        level="WARNING",
        format="{message}",
    )
    try:
        result = func()
    finally:
        logger.remove(sink_id)
    return result, [record["message"] for record in records]


class _SequenceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, out, data, model):
        return self.loss(out, data.y)


class _SecondTimestepLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.timestep = 0

    def forward(self, out, data, model):
        loss = out.sum() if self.timestep == 1 else out.sum() * 0.0
        self.timestep += 1
        return loss


class _Loader:
    def __len__(self):
        return 4


class _DummyGradScaler:
    def __init__(self, state=None):
        self._state = {"scale": 123.0} if state is None else dict(state)
        self.loaded_state = None
        self.unscale_calls = 0

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self.loaded_state = dict(state)
        self._state = dict(state)

    def unscale_(self, optimizer):
        self.unscale_calls += 1


class HMAGATCSTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_eval_step_metrics_rows_pad_to_max_episode_steps(self):
        rows = test_imitation_learning_pyg._build_step_metrics_rows(
            model_label="HMAGAT-CS",
            graph_idx=0,
            instance_seed=123,
            sampling_seed=42,
            solved_agents_per_step=[0, 2, 4],
            total_agents=4,
            max_episode_steps=5,
            rollout_success=True,
            makespan=2,
            pad_to_max_episode_steps=True,
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["success_fraction"], 0.0)
        self.assertEqual(rows[2]["success_fraction"], 1.0)
        self.assertEqual(rows[-1]["success_fraction"], 1.0)
        self.assertEqual(rows[-1]["step"], 5)
        self.assertEqual(rows[-1]["all_agents_on_goal"], 1)

    def test_eval_step_metrics_rows_no_pad_preserves_length(self):
        rows = test_imitation_learning_pyg._build_step_metrics_rows(
            model_label="MAGAT",
            graph_idx=0,
            instance_seed=123,
            sampling_seed=42,
            solved_agents_per_step=[0, 1, 2],
            total_agents=4,
            max_episode_steps=10,
            rollout_success=False,
            makespan=2,
            pad_to_max_episode_steps=False,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["step"], 2)
        self.assertEqual(rows[-1]["success_fraction"], 0.5)
        self.assertEqual(rows[-1]["rollout_success"], 0)

    def test_training_args_include_pilot_batch_limits(self):
        parser = argparse.ArgumentParser()
        add_training_args(parser)

        args = parser.parse_args([])
        self.assertIsNone(args.max_train_batches)
        self.assertIsNone(args.max_validation_batches)
        self.assertFalse(args.cs_warmup_freeze_baseline)
        self.assertIsNone(args.resume_checkpoint_path)
        self.assertIsNone(args.tensorboard_dir)
        self.assertEqual(args.tensorboard_flush_secs, 30)
        self.assertEqual(args.collision_shielding, "naive")
        self.assertEqual(args.action_sampling, "deterministic")
        self.assertEqual(args.sharded_dataset_cache_size, 1)
        self.assertEqual(args.dataloader_num_workers, 0)
        self.assertFalse(args.dataloader_pin_memory)
        self.assertFalse(args.dataloader_persistent_workers)
        self.assertIsNone(args.dataloader_prefetch_factor)
        self.assertFalse(args.amp)
        self.assertEqual(args.amp_dtype, "bfloat16")
        self.assertIsNone(args.validate_sequence_training_dataset)

        args = parser.parse_args(
            [
                "--max_train_batches",
                "7",
                "--max_validation_batches",
                "3",
                "--cs_warmup_freeze_baseline",
                "--resume_checkpoint_path",
                "resume.pt",
                "--tensorboard_dir",
                "runs/test",
                "--tensorboard_flush_secs",
                "11",
                "--dataloader_num_workers",
                "8",
                "--sharded_dataset_cache_size",
                "3",
                "--dataloader_pin_memory",
                "--dataloader_persistent_workers",
                "--dataloader_prefetch_factor",
                "4",
                "--amp",
                "--amp_dtype",
                "float16",
            ]
        )
        self.assertEqual(args.max_train_batches, 7)
        self.assertEqual(args.max_validation_batches, 3)
        self.assertTrue(args.cs_warmup_freeze_baseline)
        self.assertEqual(args.resume_checkpoint_path, "resume.pt")
        self.assertEqual(args.tensorboard_dir, "runs/test")
        self.assertEqual(args.tensorboard_flush_secs, 11)
        self.assertEqual(args.collision_shielding, "naive")
        self.assertEqual(args.action_sampling, "deterministic")
        self.assertEqual(args.sharded_dataset_cache_size, 3)
        self.assertEqual(args.dataloader_num_workers, 8)
        self.assertTrue(args.dataloader_pin_memory)
        self.assertTrue(args.dataloader_persistent_workers)
        self.assertEqual(args.dataloader_prefetch_factor, 4)
        self.assertTrue(args.amp)
        self.assertEqual(args.amp_dtype, "float16")

    def test_sharded_sequence_audit_stamp_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args, manifest = _write_tiny_processed_shard_manifest(
                tmp_dir, "audit_demo", num_samples=4
            )

            stamp_path, payload = write_sequence_audit_stamp(
                args,
                manifest,
                checked_train=True,
                checked_validation=True,
            )

            self.assertEqual(stamp_path, sequence_audit_stamp_path(args))
            self.assertTrue(os.path.exists(stamp_path))
            self.assertTrue(payload["checked_train"])
            self.assertTrue(payload["checked_validation"])

            loaded = load_sequence_audit_stamp(args)
            self.assertEqual(loaded["override_name"], "audit_demo")
            self.assertEqual(loaded["processed_entries"], payload["processed_entries"])

    def test_sharded_sequence_audit_stamp_rejects_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args, manifest = _write_tiny_processed_shard_manifest(
                tmp_dir, "audit_demo", num_samples=4
            )
            write_sequence_audit_stamp(
                args,
                manifest,
                checked_train=True,
                checked_validation=True,
            )

            manifest["entries"][0]["snapshot_count"] = 999
            with self.assertRaisesRegex(
                ValueError,
                "Sequence audit stamp processed manifest signature mismatch",
            ):
                load_sequence_audit_stamp(args, processed_manifest=manifest)

    def test_sharded_sequence_validation_mode_requires_stamp_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args, _ = _write_tiny_processed_shard_manifest(
                tmp_dir, "audit_demo", num_samples=4
            )
            args.sequence_training = True
            args.use_shards = True
            args.validate_sequence_training_dataset = None

            with self.assertRaisesRegex(
                ValueError,
                "requires a prior successful audit_sequence_dataset --use_shards run",
            ):
                _resolve_sequence_dataset_validation_mode(args)

    def test_sharded_sequence_validation_mode_accepts_valid_stamp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args, manifest = _write_tiny_processed_shard_manifest(
                tmp_dir, "audit_demo", num_samples=4
            )
            write_sequence_audit_stamp(
                args,
                manifest,
                checked_train=True,
                checked_validation=True,
            )
            args.sequence_training = True
            args.use_shards = True
            args.validate_sequence_training_dataset = None

            mode, payload = _resolve_sequence_dataset_validation_mode(args)

            self.assertEqual(mode, "audit_stamp")
            self.assertEqual(payload["override_name"], "audit_demo")

    def test_sharded_sequence_loader_risk_warnings_flag_aggressive_profile(self):
        args = SimpleNamespace(
            use_shards=True,
            sequence_training=True,
            dataloader_num_workers=12,
            dataloader_persistent_workers=True,
            dataloader_prefetch_factor=4,
        )

        warnings = _sharded_sequence_loader_risk_warnings(args)

        self.assertEqual(len(warnings), 3)
        self.assertIn("dataloader_num_workers > 0", warnings[0])
        self.assertIn("dataloader_persistent_workers", warnings[1])
        self.assertIn("dataloader_prefetch_factor", warnings[2])

    def test_build_dataloader_kwargs_rejects_prefetch_without_workers(self):
        args = SimpleNamespace(
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=2,
        )

        _, warnings = _capture_loguru_warnings(
            lambda: self.assertRaisesRegex(
                ValueError,
                "requires dataloader_num_workers > 0",
                _build_dataloader_kwargs,
                args,
            )
        )

        self.assertIn(
            "--dataloader_prefetch_factor requires dataloader_num_workers > 0",
            "\n".join(warnings),
        )

    def test_build_dataloader_kwargs_rejects_persistent_workers_without_workers(self):
        args = SimpleNamespace(
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=True,
            dataloader_prefetch_factor=None,
        )

        _, warnings = _capture_loguru_warnings(
            lambda: self.assertRaisesRegex(
                ValueError,
                "requires dataloader_num_workers > 0",
                _build_dataloader_kwargs,
                args,
            )
        )

        self.assertIn(
            "--dataloader_persistent_workers requires dataloader_num_workers > 0",
            "\n".join(warnings),
        )

    def test_build_dataloader_kwargs_include_worker_runtime_options(self):
        args = SimpleNamespace(
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            dataloader_prefetch_factor=3,
        )

        kwargs = _build_dataloader_kwargs(args)

        self.assertEqual(
            kwargs,
            {
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 3,
            },
        )

    def test_validate_amp_args_rejects_cpu_amp(self):
        args = SimpleNamespace(amp=True, amp_dtype="bfloat16")

        _, warnings = _capture_loguru_warnings(
            lambda: self.assertRaisesRegex(
                ValueError,
                "Automatic mixed precision requires a CUDA device",
                _validate_amp_args,
                args,
                torch.device("cpu"),
            )
        )

        self.assertIn(
            "Automatic mixed precision requires a CUDA device",
            "\n".join(warnings),
        )

    def test_validate_amp_args_rejects_unsupported_bfloat16(self):
        args = SimpleNamespace(amp=True, amp_dtype="bfloat16")

        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.is_bf16_supported", return_value=False
        ):
            _, warnings = _capture_loguru_warnings(
                lambda: self.assertRaisesRegex(
                    ValueError,
                    "Requested AMP dtype bfloat16 is not supported",
                    _validate_amp_args,
                    args,
                    torch.device("cuda"),
                )
            )

        self.assertIn(
            "Requested AMP dtype bfloat16 is not supported",
            "\n".join(warnings),
        )

    def test_sharded_episode_sampler_preserves_order_without_shuffle(self):
        snapshot_dataset = SimpleNamespace(episode_shard_indices=[0, 1, 0, 1])
        sampler = ShardedEpisodeSampler(snapshot_dataset, shuffle=False)
        self.assertEqual(list(iter(sampler)), [0, 1, 2, 3])

    def test_build_training_index_payload_recovers_original_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_path = os.path.join(tmp_dir, "processed.pkl")
            with open(processed_path, "wb") as f:
                pickle.dump(_tiny_dense_dataset(num_maps=2), f)

            raw_shards_dir = os.path.join(
                tmp_dir, "raw_expert_predictions", "shards"
            )
            os.makedirs(raw_shards_dir, exist_ok=True)
            raw_shard_path = os.path.join(raw_shards_dir, "raw_shard.pkl")
            with open(raw_shard_path, "wb") as f:
                pickle.dump(
                    {
                        "dataset": [("sample-0",), ("sample-2",)],
                        "seed_mask": [True, False, True, False],
                        "sample_start": 0,
                        "sample_end": 4,
                        "num_samples": 4,
                        "dataset_seed": 42,
                        "args": {},
                    },
                    f,
                )

            manifests = {
                "processed_dataset": {
                    "total_saved_samples": 2,
                    "total_snapshots": 2,
                    "entries": [
                        {
                            "path": processed_path,
                            "file_name": "processed.pkl",
                            "shard_idx": 0,
                            "sample_start": 0,
                            "sample_end": 4,
                            "saved_samples": 2,
                            "snapshot_count": 2,
                            "graph_map_id_start": 0,
                            "graph_map_id_end": 1,
                        }
                    ],
                }
            }
            args = SimpleNamespace(
                dataset_dir=tmp_dir,
                num_samples=4,
                dataset_seed=42,
                override_name="tiny",
                load_positions_separately=False,
                use_edge_attr=False,
            )

            payload = _build_training_index_payload(manifests, args)

        self.assertEqual(payload["episodes"][0]["original_sample_id"], 0)
        self.assertEqual(payload["episodes"][1]["original_sample_id"], 2)

    def test_validation_rollout_graph_ids_use_original_sample_ids_for_shards(self):
        validation_dataset = SimpleNamespace(
            episode_original_sample_ids=[101, 105, 110]
        )
        rollout_graph_ids, total_validation_graphs = _validation_rollout_graph_ids(
            use_shards=True,
            train_id_max=10,
            cur_validation_id_max=12,
            validation_id_max=13,
            validation_dataset=validation_dataset,
        )

        self.assertEqual(rollout_graph_ids, [101, 105])
        self.assertEqual(total_validation_graphs, 3)

    def test_build_sharded_dataset_state_uses_snapshot_validation_len(self):
        class _Sized:
            def __init__(self, value):
                self.value = value

            def __len__(self):
                return self.value

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_index_dir = os.path.join(tmp_dir, "processed_dataset", "shards")
            os.makedirs(training_index_dir, exist_ok=True)
            training_index_file = os.path.join(training_index_dir, "training_index.json")
            with open(training_index_file, "w") as f:
                f.write("{}")

            args = SimpleNamespace(
                dataset_dir=tmp_dir,
                override_name="tiny",
                num_samples=4,
                dataset_seed=42,
                load_positions_separately=False,
                use_edge_attr=False,
            )
            fake_manifests = {
                "processed_dataset": {
                    "total_saved_samples": 4,
                    "total_snapshots": 9,
                    "entries": [],
                }
            }
            train_dataset = _Sized(6)
            train_dataset.training_index = SimpleNamespace(manifests=fake_manifests)
            validation_dataset = _Sized(3)

            state = _build_sharded_dataset_state(
                args,
                train_dataset,
                validation_dataset=validation_dataset,
                train_id_max=2,
                validation_id_max=3,
            )

        self.assertEqual(state["train_dataset_len"], 6)
        self.assertEqual(state["validation_dataset_len"], 3)

    def test_create_tensorboard_writer_rejects_empty_dir(self):
        args = SimpleNamespace(
            tensorboard_dir="",
            tensorboard_flush_secs=30,
            checkpoints_dir="checkpoints/test",
            run_name="tb-test",
        )

        _, warnings = _capture_loguru_warnings(
            lambda: self.assertRaisesRegex(
                ValueError,
                "requires a non-empty --tensorboard_dir",
                _create_tensorboard_writer,
                args,
            )
        )

        self.assertIn(
            "requires a non-empty --tensorboard_dir",
            "\n".join(warnings),
        )

    def test_tensorboard_writer_logs_scalars_to_event_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SimpleNamespace(
                tensorboard_dir=tmp_dir,
                tensorboard_flush_secs=1,
                checkpoints_dir="checkpoints/tb-test",
                run_name="tb-test",
            )
            writer = _create_tensorboard_writer(args)
            self.addCleanup(writer.close)

            _log_tensorboard_scalars(
                writer,
                {
                    "train_loss": np.float32(1.25),
                    "validation_success_rate": np.float64(0.5),
                    "oe_improve_quality": True,
                },
                step=3,
            )
            writer.flush()
            writer.close()

            event_files = [
                os.path.join(tmp_dir, name)
                for name in os.listdir(tmp_dir)
                if name.startswith("events.out.tfevents")
            ]
            self.assertTrue(event_files)

            accumulator = event_accumulator.EventAccumulator(tmp_dir)
            accumulator.Reload()

            train_loss_events = accumulator.Scalars("train_loss")
            self.assertEqual(len(train_loss_events), 1)
            self.assertEqual(train_loss_events[0].step, 3)
            self.assertAlmostEqual(train_loss_events[0].value, 1.25, places=5)

            success_events = accumulator.Scalars("validation_success_rate")
            self.assertEqual(len(success_events), 1)
            self.assertAlmostEqual(success_events[0].value, 0.5, places=5)

            oe_events = accumulator.Scalars("oe_improve_quality")
            self.assertEqual(len(oe_events), 1)
            self.assertAlmostEqual(oe_events[0].value, 1.0, places=5)

    def test_training_batch_limit_helpers_warn_for_pilot_mode(self):
        args = SimpleNamespace(
            max_train_batches=10,
            max_validation_batches=5,
            skip_validation=False,
            skip_validation_accuracy=False,
            run_online_expert=False,
        )

        _, warnings = _capture_loguru_warnings(
            lambda: _validate_and_warn_batch_limits(args)
        )

        warning_text = "\n".join(warnings)
        self.assertIn("Limiting training to first 10 batches", warning_text)
        self.assertIn(
            "Limiting validation accuracy to first 5 batches", warning_text
        )
        self.assertEqual(_effective_num_batches(100, args.max_train_batches), 10)
        self.assertEqual(_effective_num_batches(3, args.max_train_batches), 3)
        self.assertEqual(_effective_num_batches(100, None), 100)

    def test_training_batch_limit_rejects_non_positive_values(self):
        args = SimpleNamespace(
            max_train_batches=0,
            max_validation_batches=None,
            skip_validation=False,
            skip_validation_accuracy=False,
            run_online_expert=False,
        )

        records = []
        sink_id = logger.add(
            lambda message: records.append(message.record),
            level="WARNING",
            format="{message}",
        )
        try:
            with self.assertRaises(ValueError):
                _validate_and_warn_batch_limits(args)
        finally:
            logger.remove(sink_id)

        self.assertIn(
            "--max_train_batches must be positive when set",
            "\n".join(record["message"] for record in records),
        )

    def test_validation_batch_limit_warns_when_validation_is_disabled(self):
        args = SimpleNamespace(
            max_train_batches=None,
            max_validation_batches=5,
            skip_validation=True,
            skip_validation_accuracy=False,
            run_online_expert=False,
        )

        _, warnings = _capture_loguru_warnings(
            lambda: _validate_and_warn_batch_limits(args)
        )

        self.assertIn(
            "--max_validation_batches=5 has no effect because validation",
            "\n".join(warnings),
        )

    def test_one_cycle_step_estimate_uses_train_batch_limit(self):
        args = SimpleNamespace(
            max_train_batches=7,
            num_epochs=3,
            skip_validation=True,
        )

        self.assertEqual(get_estimated_total_number_of_steps(args, _Loader()), 12)

    def test_checkpoint_source_args_reject_resume_with_other_weight_sources(self):
        args = SimpleNamespace(
            resume_checkpoint_path="resume.pt",
            pretrain_weights_path="pretrain.pt",
            load_partial_parameters_path=None,
        )

        records = []
        sink_id = logger.add(
            lambda message: records.append(message.record),
            level="WARNING",
            format="{message}",
        )
        try:
            with self.assertRaisesRegex(
                ValueError, "must not be combined"
            ):
                _validate_checkpoint_source_args(args)
        finally:
            logger.remove(sink_id)

        self.assertIn(
            "--resume_checkpoint_path must not be combined",
            "\n".join(record["message"] for record in records),
        )

    def test_coordination_state_extends_decoder_and_persists_in_simulation(self):
        model = _model(coordination_state_size=4)
        data = _graph_data()
        obs = _observations()

        self.assertEqual(model.actionsMLP[0].in_features, 12)
        self.assertIsNone(model._coordination_state)

        model.in_simulation(True)
        out_1 = model(obs, data)

        self.assertEqual(tuple(out_1.shape), (4, 5))
        self.assertIsNotNone(model._coordination_state)
        self.assertEqual(tuple(model._coordination_state.shape), (4, 4))
        self.assertFalse(model._coordination_state.requires_grad)

        state_1 = model._coordination_state.clone()
        model(obs + 0.1, data)

        self.assertFalse(torch.allclose(state_1, model._coordination_state))

        model.reset_coordination_state()
        self.assertIsNone(model._coordination_state)

    def test_snapshot_mode_does_not_store_coordination_state(self):
        model = _model(coordination_state_size=4)
        data = _graph_data()
        obs = _observations()

        out_1 = model(obs, data)
        out_2 = model(obs, data)

        self.assertIsNone(model._coordination_state)
        torch.testing.assert_close(out_1, out_2)

    def test_coordination_state_can_keep_graph_for_sequence_bptt(self):
        model = _model(coordination_state_size=4)
        data = _graph_data()
        obs_1 = _observations().requires_grad_(True)
        obs_2 = _observations()

        model.in_simulation(True)
        model.set_coordination_state_detach(False)
        model(obs_1, data)

        self.assertTrue(model._coordination_state.requires_grad)

        out_2 = model(obs_2, data)
        loss = out_2.sum()
        loss.backward()

        self.assertIsNotNone(obs_1.grad)
        self.assertGreater(torch.norm(obs_1.grad).item(), 0.0)

    def test_cnn_to_out_residual_keeps_pre_coordination_state_size(self):
        model = _model(coordination_state_size=4, module_residual=["cnn-to-out"])
        data = _graph_data()
        obs = _observations()

        self.assertEqual(model.cnn_to_out_lin.out_features, 8)
        self.assertEqual(model.actionsMLP[0].in_features, 12)
        self.assertEqual(tuple(model(obs, data).shape), (4, 5))

    def test_directional_hmagat_coordination_state_forward(self):
        model = _model(
            coordination_state_size=4,
            gnn_type="DirectionalHMAGAT",
            gnn_kwargs={
                "hyperedge_feature_generator": "magat",
                "final_feature_generator": "magat",
            },
        )
        data = _hypergraph_data()
        obs = _observations()

        model.in_simulation(True)
        out = model(obs, data)

        self.assertEqual(tuple(out.shape), (4, 5))
        self.assertEqual(tuple(model._coordination_state.shape), (4, 4))

    def test_coordination_state_gru_keeps_internal_state_fp32_for_low_precision_input(self):
        model = _model(coordination_state_size=4)
        model.in_simulation(True)

        x = torch.randn(4, 8, dtype=torch.bfloat16)
        out = model._apply_coordination_state(x)

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (4, 12))
        self.assertIsNotNone(model._coordination_state)
        self.assertEqual(model._coordination_state.dtype, torch.float32)

    def test_coordination_state_gru_runs_under_cuda_bfloat16_autocast(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for AMP autocast test.")
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA bfloat16 support is required for AMP autocast test.")

        model = _model(coordination_state_size=4).to("cuda")
        model.in_simulation(True)
        x = torch.randn(4, 8, device="cuda", dtype=torch.float32)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model._apply_coordination_state(x)

        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(tuple(out.shape), (4, 12))
        self.assertIsNotNone(model._coordination_state)
        self.assertEqual(model._coordination_state.dtype, torch.float32)

    def test_directional_hmagat_forward_runs_under_cuda_bfloat16_autocast(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for AMP autocast test.")
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA bfloat16 support is required for AMP autocast test.")

        model = _model(
            coordination_state_size=4,
            gnn_type="DirectionalHMAGAT",
            gnn_kwargs={
                "hyperedge_feature_generator": "magat",
                "final_feature_generator": "magat",
            },
        ).to("cuda")
        data = _hypergraph_data().to("cuda")
        obs = _observations().to("cuda")

        model.in_simulation(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(obs, data)

        self.assertEqual(tuple(out.shape), (4, 5))
        self.assertEqual(out.device.type, "cuda")
        self.assertIsNotNone(model._coordination_state)
        self.assertEqual(model._coordination_state.dtype, torch.float32)

    def test_partial_state_loading_widens_decoder_layer_with_zero_extra_columns(self):
        old_model = _model(coordination_state_size=0)
        new_model = _model(coordination_state_size=4)
        old_state = old_model.state_dict()

        for key, value in old_state.items():
            if torch.is_floating_point(value):
                old_state[key] = torch.full_like(value, 0.25)

        summary, warnings = _capture_loguru_warnings(
            lambda: load_partial_state_dict(new_model, old_state, print_prefix="[test]")
        )

        self.assertIn("cnn.convs.0.weight", summary["loaded"])
        self.assertIn("actionsMLP.0.weight", summary["loaded"])
        self.assertIn("actionsMLP.0.bias", summary["loaded"])
        self.assertIn("actionsMLP.0.weight", summary["widened_linear"])
        self.assertEqual(summary["skipped_shape"], [])
        self.assertEqual(summary["skipped_related"], [])
        self.assertIn(
            "coordination_state_cell.weight_ih", summary["missing_after_load"]
        )
        self.assertNotIn("actionsMLP.0.weight", summary["missing_after_load"])
        self.assertNotIn("actionsMLP.0.bias", summary["missing_after_load"])

        torch.testing.assert_close(
            new_model.state_dict()["cnn.convs.0.weight"],
            torch.full_like(new_model.state_dict()["cnn.convs.0.weight"], 0.25),
        )
        torch.testing.assert_close(
            new_model.state_dict()["actionsMLP.0.weight"][:, :8],
            torch.full_like(new_model.state_dict()["actionsMLP.0.weight"][:, :8], 0.25),
        )
        torch.testing.assert_close(
            new_model.state_dict()["actionsMLP.0.weight"][:, 8:],
            torch.zeros_like(new_model.state_dict()["actionsMLP.0.weight"][:, 8:]),
        )
        torch.testing.assert_close(
            new_model.state_dict()["actionsMLP.0.bias"],
            torch.full_like(new_model.state_dict()["actionsMLP.0.bias"], 0.25),
        )
        self.assertTrue(
            any("widened linear keys: 1" in message for message in warnings)
        )
        self.assertTrue(
            any("coordination_state_cell.weight_ih" in message for message in warnings)
        )

    def test_partial_state_loading_preserves_baseline_logits_with_zero_extra_columns(self):
        old_model = _model(coordination_state_size=0)
        new_model = _model(coordination_state_size=4)
        data = _graph_data()
        obs = _observations()

        load_partial_state_dict(new_model, old_model.state_dict(), print_prefix="[test]")

        with torch.no_grad():
            old_out = old_model(obs, data)
            new_out = new_model(obs, data)

        torch.testing.assert_close(new_out, old_out)

    def test_baseline_checkpoint_is_compatible_with_resnet_hmagat_cs_config(self):
        checkpoint_path = os.path.join("checkpoints", "hmagat", "best.pt")
        self.assertTrue(
            os.path.exists(checkpoint_path),
            "Expected local pretrained HMAGAT checkpoint for compatibility test.",
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model, _, _ = get_model(
            _checkpoint_compatible_args(),
            torch.device("cpu"),
        )

        summary, _warnings = _capture_loguru_warnings(
            lambda: load_partial_state_dict(
                model, state_dict, print_prefix="[checkpoint-test]"
            )
        )

        validate_partial_load_compatibility(
            summary, print_prefix="[checkpoint-test]"
        )
        self.assertEqual(summary["skipped_missing"], [])
        self.assertEqual(summary["skipped_shape"], [])
        self.assertIn("actionsMLP.0.weight", summary["widened_linear"])

    def test_baseline_checkpoint_rejects_wrong_cnn_mode_for_hmagat_cs(self):
        checkpoint_path = os.path.join("checkpoints", "hmagat", "best.pt")
        self.assertTrue(
            os.path.exists(checkpoint_path),
            "Expected local pretrained HMAGAT checkpoint for compatibility test.",
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model, _, _ = get_model(
            _checkpoint_compatible_args(cnn_mode="basic-CNN"),
            torch.device("cpu"),
        )
        summary, _warnings = _capture_loguru_warnings(
            lambda: load_partial_state_dict(
                model, state_dict, print_prefix="[checkpoint-test]"
            )
        )

        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_partial_load_compatibility(
                summary, print_prefix="[checkpoint-test]"
            )

    def test_cs_warmup_freeze_requires_widened_decoder_partial_load(self):
        args = SimpleNamespace(
            cs_warmup_freeze_baseline=True,
            coordination_state_size=4,
            load_partial_parameters_path="checkpoints/hmagat/best.pt",
        )
        summary = {"widened_linear": []}

        records = []
        sink_id = logger.add(
            lambda message: records.append(message.record),
            level="WARNING",
            format="{message}",
        )
        try:
            with self.assertRaisesRegex(
                ValueError, "requires widened partial loading"
            ):
                _validate_cs_warmup_freeze_baseline(args, summary)
        finally:
            logger.remove(sink_id)

        self.assertIn(
            "--cs_warmup_freeze_baseline requires widened partial loading for "
            "actionsMLP.0.weight",
            "\n".join(record["message"] for record in records),
        )

    def test_cs_warmup_freeze_leaves_only_coordination_cell_and_decoder_weight_trainable(self):
        baseline_model = _model(coordination_state_size=0)
        model = _model(coordination_state_size=4)
        summary = load_partial_state_dict(
            model, baseline_model.state_dict(), print_prefix="[test]"
        )
        args = SimpleNamespace(
            cs_warmup_freeze_baseline=True,
            coordination_state_size=4,
            load_partial_parameters_path="checkpoints/hmagat/best.pt",
        )

        _apply_cs_warmup_freeze_baseline(model, args, summary)

        trainable = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        self.assertEqual(
            trainable,
            {
                "actionsMLP.0.weight",
                "coordination_state_cell.weight_ih",
                "coordination_state_cell.weight_hh",
                "coordination_state_cell.bias_ih",
                "coordination_state_cell.bias_hh",
            },
        )
        self.assertFalse(model.actionsMLP[0].bias.requires_grad)
        self.assertFalse(model.actionsMLP[1].weight.requires_grad)
        self.assertFalse(model.actionsMLP[1].bias.requires_grad)

    def test_cs_warmup_freeze_masks_old_decoder_gradients_and_restores_prefix(self):
        baseline_model = _model(coordination_state_size=0)
        model = _model(coordination_state_size=4)
        summary = load_partial_state_dict(
            model, baseline_model.state_dict(), print_prefix="[test]"
        )
        args = SimpleNamespace(
            cs_warmup_freeze_baseline=True,
            coordination_state_size=4,
            load_partial_parameters_path="checkpoints/hmagat/best.pt",
            lr_start=1e-3,
            weight_decay=1e-2,
        )

        warmup = _apply_cs_warmup_freeze_baseline(model, args, summary)
        optimizer = _create_optimizer(args, model, warmup)
        decoder_weight_group = next(
            group
            for group in optimizer.param_groups
            if any(
                param is model.actionsMLP[0].weight for param in group["params"]
            )
        )
        self.assertEqual(decoder_weight_group["weight_decay"], 0.0)

        data = _graph_data()
        obs = _observations()
        model.train()
        before_new_columns = model.actionsMLP[0].weight[
            :, warmup.old_decoder_input_size :
        ].detach().clone()

        loss = model(obs, data).sum()
        loss.backward()

        grad = model.actionsMLP[0].weight.grad
        self.assertIsNotNone(grad)
        torch.testing.assert_close(
            grad[:, : warmup.old_decoder_input_size],
            torch.zeros_like(grad[:, : warmup.old_decoder_input_size]),
        )
        self.assertGreater(
            torch.norm(grad[:, warmup.old_decoder_input_size :]).item(),
            0.0,
        )

        with torch.no_grad():
            model.actionsMLP[0].weight[:, : warmup.old_decoder_input_size].add_(1.0)
        warmup.restore_decoder_prefix()
        torch.testing.assert_close(
            model.actionsMLP[0].weight[:, : warmup.old_decoder_input_size],
            warmup.decoder_prefix_snapshot,
        )

        optimizer.step()
        warmup.restore_decoder_prefix()
        torch.testing.assert_close(
            model.actionsMLP[0].weight[:, : warmup.old_decoder_input_size],
            warmup.decoder_prefix_snapshot,
        )
        self.assertFalse(
            torch.allclose(
                model.actionsMLP[0].weight[:, warmup.old_decoder_input_size :],
                before_new_columns,
            )
        )

    def test_extract_model_state_dict_accepts_training_checkpoint_payload(self):
        model = _model()
        checkpoint = {
            "checkpoint_type": TRAINING_CHECKPOINT_TYPE,
            "model_state_dict": model.state_dict(),
        }

        state_dict, warnings = _capture_loguru_warnings(
            lambda: extract_model_state_dict_from_checkpoint(
                checkpoint, "checkpoint.pt", print_prefix="[test]"
            )
        )

        torch.testing.assert_close(
            state_dict["actionsMLP.0.weight"],
            model.state_dict()["actionsMLP.0.weight"],
        )
        self.assertTrue(
            any("Training checkpoint payload detected" in message for message in warnings)
        )

    def test_resume_checkpoint_rejects_legacy_model_only_state_dict(self):
        model = _model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "legacy.pt")
            torch.save(model.state_dict(), checkpoint_path)

            _, warnings = _capture_loguru_warnings(
                lambda: self.assertRaisesRegex(
                    ValueError,
                    "requires a training checkpoint payload",
                    _load_resume_training_checkpoint,
                    checkpoint_path,
                    map_location="cpu",
                )
            )

        self.assertTrue(
            any("requires a training checkpoint payload" in message for message in warnings)
        )

    def test_resume_checkpoint_rejects_missing_dataset_and_rng_state(self):
        model = _model()
        checkpoint = {
            "format_version": TRAINING_CHECKPOINT_VERSION,
            "checkpoint_type": TRAINING_CHECKPOINT_TYPE,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "lr_scheduler_state_dict": {},
            "epoch": 0,
            "args": {},
            "training_state": {
                "best_validation_success_rate": 0.0,
                "best_validation_accuracy": 0.0,
                "best_val_file_name": "best_low_val.pt",
                "cur_validation_id_max": 0,
                "threshold_val_success_rate": 0.9,
                "oe_improve_quality": False,
                "cs_warmup_freeze_baseline": False,
                "cs_warmup_old_decoder_input_size": None,
                "cs_warmup_decoder_prefix_snapshot": None,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "resume_missing_state.pt")
            torch.save(checkpoint, checkpoint_path)

            _, warnings = _capture_loguru_warnings(
                lambda: self.assertRaisesRegex(
                    ValueError,
                    "missing required keys",
                    _load_resume_training_checkpoint,
                    checkpoint_path,
                    map_location="cpu",
                )
            )

        self.assertTrue(any("missing required keys" in message for message in warnings))

    def test_resume_checkpoint_rejects_dataset_binding_mismatch(self):
        saved_dataset_state = {
            "mode": "unsharded",
            "train_id_max": 10,
            "validation_id_max": 15,
            "train_dataset_len": 100,
            "validation_dataset_len": 20,
            "stages": {
                "processed_dataset": [
                    {"path": "/tmp/a.pkl", "size": 1, "mtime_ns": 1}
                ]
            },
        }
        current_dataset_state = {
            "mode": "unsharded",
            "train_id_max": 10,
            "validation_id_max": 15,
            "train_dataset_len": 100,
            "validation_dataset_len": 20,
            "stages": {
                "processed_dataset": [
                    {"path": "/tmp/b.pkl", "size": 1, "mtime_ns": 1}
                ]
            },
        }

        _, warnings = _capture_loguru_warnings(
            lambda: self.assertRaisesRegex(
                ValueError,
                "dataset binding mismatch",
                _validate_resume_dataset_state,
                current_dataset_state,
                saved_dataset_state,
            )
        )

        self.assertTrue(any("dataset binding mismatch" in message for message in warnings))

    def test_resume_checkpoint_args_reject_max_train_batches_change(self):
        resume_checkpoint = {
            "epoch": 0,
            "args": {
                "run_online_expert": False,
                "cs_warmup_freeze_baseline": False,
                "max_train_batches": None,
                "max_validation_batches": None,
                "num_epochs": 3,
            },
            "training_state": {
                "cs_warmup_freeze_baseline": False,
            },
        }
        args = SimpleNamespace(
            cs_warmup_freeze_baseline=False,
            max_train_batches=5,
            max_validation_batches=None,
            num_epochs=3,
            run_online_expert=False,
        )

        _, warnings = _capture_loguru_warnings(
            lambda: self.assertRaisesRegex(
                ValueError,
                "arguments are incompatible",
                _validate_resume_checkpoint_args,
                args,
                resume_checkpoint,
            )
        )

        self.assertTrue(any("max_train_batches" in message for message in warnings))

    def test_resume_checkpoint_args_allow_num_epochs_extension(self):
        resume_checkpoint = {
            "epoch": 1,
            "args": {
                "run_online_expert": False,
                "cs_warmup_freeze_baseline": False,
                "max_train_batches": 5,
                "max_validation_batches": 2,
                "num_epochs": 2,
            },
            "training_state": {
                "cs_warmup_freeze_baseline": False,
            },
        }
        args = SimpleNamespace(
            cs_warmup_freeze_baseline=False,
            max_train_batches=5,
            max_validation_batches=2,
            num_epochs=4,
            run_online_expert=False,
        )

        _validate_resume_checkpoint_args(args, resume_checkpoint)

    def test_resume_checkpoint_args_allow_dataloader_and_amp_changes(self):
        resume_checkpoint = {
            "epoch": 1,
            "args": {
                "run_online_expert": False,
                "cs_warmup_freeze_baseline": False,
                "max_train_batches": 5,
                "max_validation_batches": 2,
                "num_epochs": 2,
                "dataloader_num_workers": 0,
                "dataloader_pin_memory": False,
                "dataloader_persistent_workers": False,
                "dataloader_prefetch_factor": None,
                "amp": False,
                "amp_dtype": "bfloat16",
            },
            "training_state": {
                "cs_warmup_freeze_baseline": False,
            },
        }
        args = SimpleNamespace(
            cs_warmup_freeze_baseline=False,
            max_train_batches=5,
            max_validation_batches=2,
            num_epochs=4,
            run_online_expert=False,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            dataloader_prefetch_factor=4,
            amp=True,
            amp_dtype="float16",
        )

        _validate_resume_checkpoint_args(args, resume_checkpoint)

    def test_resume_checkpoint_args_reject_num_epochs_decrease(self):
        resume_checkpoint = {
            "epoch": 0,
            "args": {
                "run_online_expert": False,
                "cs_warmup_freeze_baseline": False,
                "max_train_batches": 5,
                "max_validation_batches": 2,
                "num_epochs": 4,
            },
            "training_state": {
                "cs_warmup_freeze_baseline": False,
            },
        }
        args = SimpleNamespace(
            cs_warmup_freeze_baseline=False,
            max_train_batches=5,
            max_validation_batches=2,
            num_epochs=2,
            run_online_expert=False,
        )

        _, warnings = _capture_loguru_warnings(
            lambda: self.assertRaisesRegex(
                ValueError,
                "num_epochs cannot be reduced",
                _validate_resume_checkpoint_args,
                args,
                resume_checkpoint,
            )
        )

        self.assertTrue(any("num_epochs cannot be reduced" in message for message in warnings))

    def test_resume_checkpoint_roundtrip_restores_optimizer_scheduler_rng_and_state(self):
        model = _model(coordination_state_size=4)
        args = SimpleNamespace(
            lr_start=1e-3,
            lr_end=1e-5,
            weight_decay=1e-4,
            lr_scheduler="one-cycle",
            num_epochs=3,
            skip_validation=True,
            max_train_batches=None,
            validation_every_epochs=1,
            run_oe_after=0,
            num_run_oe=1,
            batch_size=2,
            run_online_expert=False,
            resume_checkpoint_path=None,
            pretrain_weights_path=None,
            load_partial_parameters_path=None,
            checkpoints_dir="checkpoints/original",
            run_name="original",
            wandb_project="proj",
            wandb_entity=None,
            max_validation_batches=None,
            device=None,
            save_intmd_checkpoints=True,
            cs_warmup_freeze_baseline=False,
            tensorboard_dir="runs/original",
            tensorboard_flush_secs=30,
        )
        optimizer = _create_optimizer(args, model, None)
        from hmagat.lr_scheduler import get_lr_scheduler

        lr_scheduler = get_lr_scheduler(args, optimizer, _Loader())
        data = _graph_data()
        obs = _observations()
        torch.manual_seed(2024)
        np.random.seed(2025)
        random.seed(2026)
        model.train()
        loss = model(obs, data).sum()
        loss.backward()
        optimizer.step()
        lr_scheduler.step_on_batch()

        payload = _build_training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=1,
            args=args,
            best_validation_success_rate=0.25,
            best_validation_accuracy=0.5,
            best_val_file_name="best_low_val.pt",
            cur_validation_id_max=123,
            threshold_val_success_rate=0.9,
            oe_improve_quality=False,
            dataset_state={
                "mode": "unit-test",
                "train_id_max": 2,
                "validation_id_max": 3,
                "train_dataset_len": 4,
                "validation_dataset_len": 1,
                "stages": {
                    "processed_dataset": [
                        {"path": "/tmp/test.pkl", "size": 1, "mtime_ns": 1}
                    ]
                },
            },
            cs_warmup_freeze_state=None,
        )
        expected_torch = torch.rand(4)
        expected_numpy = np.random.rand(3)
        expected_python = [random.random() for _ in range(3)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "resume.pt")
            torch.save(payload, checkpoint_path)
            loaded_payload = _load_resume_training_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )

        resume_args = SimpleNamespace(**vars(args))
        resume_args.resume_checkpoint_path = "resume.pt"
        resume_args.tensorboard_dir = "runs/resume"
        resume_args.tensorboard_flush_secs = 5
        _validate_resume_checkpoint_args(resume_args, loaded_payload)

        restored_model = _model(coordination_state_size=4)
        restored_optimizer = _create_optimizer(resume_args, restored_model, None)
        restored_scheduler = get_lr_scheduler(
            resume_args, restored_optimizer, _Loader()
        )
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        resume_state = _restore_resume_training_state(
            restored_model,
            restored_optimizer,
            restored_scheduler,
            loaded_payload,
        )

        self.assertEqual(resume_state["start_epoch"], 2)
        self.assertEqual(
            resume_state["best_validation_success_rate"], 0.25
        )
        self.assertEqual(resume_state["best_validation_accuracy"], 0.5)
        self.assertEqual(resume_state["best_val_file_name"], "best_low_val.pt")
        self.assertEqual(resume_state["cur_validation_id_max"], 123)
        self.assertEqual(resume_state["threshold_val_success_rate"], 0.9)
        self.assertFalse(resume_state["oe_improve_quality"])
        torch.testing.assert_close(
            restored_model.state_dict()["actionsMLP.0.weight"],
            model.state_dict()["actionsMLP.0.weight"],
        )
        self.assertEqual(
            restored_scheduler.state_dict()["cur_step"],
            lr_scheduler.state_dict()["cur_step"],
        )
        self.assertEqual(
            len(restored_optimizer.state_dict()["state"]),
            len(optimizer.state_dict()["state"]),
        )
        torch.testing.assert_close(torch.rand(4), expected_torch)
        np.testing.assert_allclose(np.random.rand(3), expected_numpy)
        self.assertEqual([random.random() for _ in range(3)], expected_python)

    def test_resolve_evaluation_checkpoint_prefers_best_file_recorded_in_last_checkpoint(self):
        model = _model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = os.path.join(tmp_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            stale_best_path = os.path.join(checkpoint_dir, "best.pt")
            expected_path = os.path.join(checkpoint_dir, "best_low_val.pt")
            torch.save(model.state_dict(), stale_best_path)
            torch.save(model.state_dict(), expected_path)
            torch.save(
                {
                    "format_version": TRAINING_CHECKPOINT_VERSION,
                    "checkpoint_type": TRAINING_CHECKPOINT_TYPE,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": {},
                    "lr_scheduler_state_dict": {},
                    "epoch": 0,
                    "args": {"num_epochs": 1},
                    "rng_state": {
                        "torch_rng_state": torch.get_rng_state(),
                        "numpy_random_state": np.random.get_state(),
                        "python_random_state": random.getstate(),
                        "cuda_rng_state_all": None,
                    },
                    "dataset_state": {
                        "mode": "unit-test",
                        "train_id_max": 1,
                        "validation_id_max": 1,
                        "train_dataset_len": 1,
                        "validation_dataset_len": 0,
                        "stages": {},
                    },
                    "training_state": {
                        "best_validation_success_rate": 0.0,
                        "best_validation_accuracy": 0.0,
                        "best_val_file_name": "best_low_val.pt",
                        "cur_validation_id_max": 0,
                        "threshold_val_success_rate": 0.9,
                        "oe_improve_quality": False,
                        "cs_warmup_freeze_baseline": False,
                        "cs_warmup_old_decoder_input_size": None,
                        "cs_warmup_decoder_prefix_snapshot": None,
                    },
                },
                os.path.join(checkpoint_dir, "last.pt"),
            )

            resolved = resolve_evaluation_checkpoint_path(
                checkpoint_dir,
                model_epoch_num=None,
                map_location="cpu",
            )

        self.assertEqual(resolved, pathlib.Path(expected_path))

    def test_partial_checkpoint_helper_rejects_incompatible_model(self):
        checkpoint_model = _model()
        target_model = _model(cnn_mode="ResNetLarge_withMLP")

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "partial.pt")
            torch.save(checkpoint_model.state_dict(), checkpoint_path)

            _, warnings = _capture_loguru_warnings(
                lambda: self.assertRaisesRegex(
                    ValueError,
                    "Partial checkpoint is incompatible",
                    load_partial_checkpoint_into_model,
                    target_model,
                    checkpoint_path,
                    map_location="cpu",
                    print_prefix="[eval-partial] ",
                )
            )

        self.assertTrue(
            any("Partial checkpoint compatibility check failed" in message for message in warnings)
        )

    def test_training_checkpoint_payload_includes_grad_scaler_state(self):
        model = _model()
        args = SimpleNamespace(
            cs_warmup_freeze_baseline=False,
            checkpoints_dir="checkpoints",
            tensorboard_dir=None,
            tensorboard_flush_secs=30,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        from hmagat.lr_scheduler import get_lr_scheduler

        scheduler_args = SimpleNamespace(
            lr_start=1e-3,
            lr_end=1e-5,
            weight_decay=1e-5,
            lr_scheduler="cosine-annealing",
            num_epochs=1,
            skip_validation=True,
            max_train_batches=None,
            validation_every_epochs=1,
            run_oe_after=0,
            num_run_oe=1,
            batch_size=1,
        )
        lr_scheduler = get_lr_scheduler(scheduler_args, optimizer, _Loader())
        grad_scaler = _DummyGradScaler({"scale": 2048.0, "growth_factor": 2.0})

        payload = _build_training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=0,
            args=args,
            best_validation_success_rate=0.0,
            best_validation_accuracy=0.0,
            best_val_file_name="best_low_val.pt",
            cur_validation_id_max=0,
            threshold_val_success_rate=0.9,
            oe_improve_quality=False,
            dataset_state={
                "mode": "unit-test",
                "train_id_max": 1,
                "validation_id_max": 1,
                "train_dataset_len": 1,
                "validation_dataset_len": 0,
                "stages": {},
            },
            cs_warmup_freeze_state=None,
            grad_scaler=grad_scaler,
        )

        self.assertEqual(
            payload["grad_scaler_state_dict"],
            {"scale": 2048.0, "growth_factor": 2.0},
        )

    def test_restore_resume_training_state_restores_grad_scaler_state(self):
        model = _model()
        args = SimpleNamespace(
            lr_start=1e-3,
            lr_end=1e-5,
            weight_decay=1e-4,
            lr_scheduler="cosine-annealing",
            num_epochs=2,
            skip_validation=True,
            max_train_batches=None,
            validation_every_epochs=1,
            run_oe_after=0,
            num_run_oe=1,
            batch_size=2,
            run_online_expert=False,
            resume_checkpoint_path=None,
            pretrain_weights_path=None,
            load_partial_parameters_path=None,
            checkpoints_dir="checkpoints/original",
            run_name="original",
            wandb_project="proj",
            wandb_entity=None,
            max_validation_batches=None,
            device=None,
            save_intmd_checkpoints=True,
            cs_warmup_freeze_baseline=False,
            tensorboard_dir=None,
            tensorboard_flush_secs=30,
        )
        optimizer = _create_optimizer(args, model, None)
        from hmagat.lr_scheduler import get_lr_scheduler

        lr_scheduler = get_lr_scheduler(args, optimizer, _Loader())
        payload = _build_training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=0,
            args=args,
            best_validation_success_rate=0.0,
            best_validation_accuracy=0.0,
            best_val_file_name="best_low_val.pt",
            cur_validation_id_max=0,
            threshold_val_success_rate=0.9,
            oe_improve_quality=False,
            dataset_state={
                "mode": "unit-test",
                "train_id_max": 1,
                "validation_id_max": 1,
                "train_dataset_len": 1,
                "validation_dataset_len": 0,
                "stages": {},
            },
            cs_warmup_freeze_state=None,
            grad_scaler=_DummyGradScaler({"scale": 1024.0}),
        )
        restored_model = _model()
        restored_optimizer = _create_optimizer(args, restored_model, None)
        restored_scheduler = get_lr_scheduler(args, restored_optimizer, _Loader())
        restored_grad_scaler = _DummyGradScaler({"scale": 1.0})

        _restore_resume_training_state(
            restored_model,
            restored_optimizer,
            restored_scheduler,
            payload,
            grad_scaler=restored_grad_scaler,
        )

        self.assertEqual(restored_grad_scaler.loaded_state, {"scale": 1024.0})

    def test_apply_gradient_clipping_clips_norm_and_unscales_amp(self):
        model = _model()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 10.0)
        args = SimpleNamespace(
            grad_clip_value=0.5,
            grad_clip_norm="2.0",
            grad_clip_type="norm",
            grad_clip_warmup_steps=None,
        )
        grad_scaler = _DummyGradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        returned_norm = _apply_gradient_clipping(
            args,
            model,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            global_step=0,
        )

        self.assertIsNotNone(returned_norm)
        self.assertEqual(grad_scaler.unscale_calls, 1)
        clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        self.assertLessEqual(float(clipped_norm), 0.500001)

    def test_restore_rng_state_normalizes_cuda_rng_tensors(self):
        rng_state = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "cuda_rng_state_all": [torch.arange(8, dtype=torch.int64)],
        }

        captured = {}

        def _capture_cuda_rng_state_all(states):
            captured["states"] = states

        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch(
                "torch.cuda.set_rng_state_all", side_effect=_capture_cuda_rng_state_all
            ):
                _restore_rng_state(rng_state)

        self.assertIn("states", captured)
        self.assertEqual(len(captured["states"]), 1)
        restored_state = captured["states"][0]
        self.assertTrue(torch.is_tensor(restored_state))
        self.assertEqual(restored_state.device.type, "cpu")
        self.assertEqual(restored_state.dtype, torch.uint8)

    def test_resume_checkpoint_roundtrip_extends_cosine_scheduler_horizon(self):
        model = _model(coordination_state_size=4)
        args = SimpleNamespace(
            lr_start=1e-3,
            lr_end=1e-5,
            weight_decay=1e-4,
            lr_scheduler="cosine-annealing",
            num_epochs=2,
            skip_validation=True,
            max_train_batches=None,
            validation_every_epochs=1,
            run_oe_after=0,
            num_run_oe=1,
            batch_size=2,
            run_online_expert=False,
            resume_checkpoint_path=None,
            pretrain_weights_path=None,
            load_partial_parameters_path=None,
            checkpoints_dir="checkpoints/original",
            run_name="original",
            wandb_project="proj",
            wandb_entity=None,
            max_validation_batches=None,
            device=None,
            save_intmd_checkpoints=True,
            cs_warmup_freeze_baseline=False,
            tensorboard_dir="runs/original",
            tensorboard_flush_secs=30,
        )
        optimizer = _create_optimizer(args, model, None)
        from hmagat.lr_scheduler import get_lr_scheduler

        lr_scheduler = get_lr_scheduler(args, optimizer, _Loader())
        for _ in range(2):
            optimizer.step()
            lr_scheduler.step_on_epoch()

        payload = _build_training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=1,
            args=args,
            best_validation_success_rate=0.25,
            best_validation_accuracy=0.5,
            best_val_file_name="best_low_val.pt",
            cur_validation_id_max=123,
            threshold_val_success_rate=0.9,
            oe_improve_quality=False,
            dataset_state={
                "mode": "unit-test",
                "train_id_max": 2,
                "validation_id_max": 3,
                "train_dataset_len": 4,
                "validation_dataset_len": 1,
                "stages": {
                    "processed_dataset": [
                        {"path": "/tmp/test.pkl", "size": 1, "mtime_ns": 1}
                    ]
                },
            },
            cs_warmup_freeze_state=None,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "resume.pt")
            torch.save(payload, checkpoint_path)
            loaded_payload = _load_resume_training_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )

        resume_args = SimpleNamespace(**vars(args))
        resume_args.num_epochs = 4
        resume_args.resume_checkpoint_path = "resume.pt"
        resume_args.tensorboard_dir = "runs/resume"
        resume_args.tensorboard_flush_secs = 5
        _validate_resume_checkpoint_args(resume_args, loaded_payload)

        restored_model = _model(coordination_state_size=4)
        restored_optimizer = _create_optimizer(resume_args, restored_model, None)
        restored_scheduler = get_lr_scheduler(
            resume_args, restored_optimizer, _Loader()
        )
        resume_state = _restore_resume_training_state(
            restored_model,
            restored_optimizer,
            restored_scheduler,
            loaded_payload,
        )

        expected_model = _model(coordination_state_size=4)
        expected_optimizer = _create_optimizer(resume_args, expected_model, None)
        expected_scheduler = get_lr_scheduler(
            resume_args, expected_optimizer, _Loader()
        )
        expected_optimizer.step()
        expected_scheduler.step_on_epoch()
        expected_optimizer.step()
        expected_scheduler.step_on_epoch()

        self.assertEqual(resume_state["start_epoch"], 2)
        self.assertEqual(
            restored_scheduler.scheduler.state_dict()["T_max"],
            expected_scheduler.scheduler.state_dict()["T_max"],
        )
        self.assertEqual(
            restored_scheduler.scheduler.state_dict()["last_epoch"],
            expected_scheduler.scheduler.state_dict()["last_epoch"],
        )
        self.assertEqual(
            restored_scheduler.state_dict()["cur_step"],
            expected_scheduler.state_dict()["cur_step"],
        )
        self.assertAlmostEqual(
            restored_optimizer.param_groups[0]["lr"],
            expected_optimizer.param_groups[0]["lr"],
            places=12,
        )

    def test_resume_checkpoint_roundtrip_extends_onecycle_scheduler_horizon(self):
        model = _model(coordination_state_size=4)
        args = SimpleNamespace(
            lr_start=1e-3,
            lr_end=1e-5,
            weight_decay=1e-4,
            lr_scheduler="one-cycle",
            num_epochs=2,
            skip_validation=True,
            max_train_batches=2,
            validation_every_epochs=1,
            run_oe_after=0,
            num_run_oe=1,
            batch_size=2,
            run_online_expert=False,
            resume_checkpoint_path=None,
            pretrain_weights_path=None,
            load_partial_parameters_path=None,
            checkpoints_dir="checkpoints/original",
            run_name="original",
            wandb_project="proj",
            wandb_entity=None,
            max_validation_batches=None,
            device=None,
            save_intmd_checkpoints=True,
            cs_warmup_freeze_baseline=False,
            tensorboard_dir="runs/original",
            tensorboard_flush_secs=30,
        )
        optimizer = _create_optimizer(args, model, None)
        from hmagat.lr_scheduler import get_lr_scheduler

        lr_scheduler = get_lr_scheduler(args, optimizer, _Loader())
        for _ in range(2):
            optimizer.step()
            lr_scheduler.step_on_batch()

        payload = _build_training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=0,
            args=args,
            best_validation_success_rate=0.25,
            best_validation_accuracy=0.5,
            best_val_file_name="best_low_val.pt",
            cur_validation_id_max=123,
            threshold_val_success_rate=0.9,
            oe_improve_quality=False,
            dataset_state={
                "mode": "unit-test",
                "train_id_max": 2,
                "validation_id_max": 3,
                "train_dataset_len": 4,
                "validation_dataset_len": 1,
                "stages": {
                    "processed_dataset": [
                        {"path": "/tmp/test.pkl", "size": 1, "mtime_ns": 1}
                    ]
                },
            },
            cs_warmup_freeze_state=None,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "resume.pt")
            torch.save(payload, checkpoint_path)
            loaded_payload = _load_resume_training_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )

        resume_args = SimpleNamespace(**vars(args))
        resume_args.num_epochs = 4
        _validate_resume_checkpoint_args(resume_args, loaded_payload)

        restored_model = _model(coordination_state_size=4)
        restored_optimizer = _create_optimizer(resume_args, restored_model, None)
        restored_scheduler = get_lr_scheduler(
            resume_args, restored_optimizer, _Loader()
        )
        resume_state = _restore_resume_training_state(
            restored_model,
            restored_optimizer,
            restored_scheduler,
            loaded_payload,
        )

        expected_model = _model(coordination_state_size=4)
        expected_optimizer = _create_optimizer(resume_args, expected_model, None)
        expected_scheduler = get_lr_scheduler(
            resume_args, expected_optimizer, _Loader()
        )
        expected_scheduler.scheduler.step(2)
        expected_scheduler.scheduler._step_count = 3
        expected_scheduler.cur_step = 2

        self.assertEqual(resume_state["start_epoch"], 1)
        self.assertEqual(
            restored_scheduler.state_dict()["max_steps"],
            expected_scheduler.state_dict()["max_steps"],
        )
        self.assertEqual(
            restored_scheduler.state_dict()["cur_step"],
            expected_scheduler.state_dict()["cur_step"],
        )
        self.assertEqual(
            restored_scheduler.scheduler.state_dict()["total_steps"],
            expected_scheduler.scheduler.state_dict()["total_steps"],
        )
        self.assertEqual(
            restored_scheduler.scheduler.state_dict()["last_epoch"],
            expected_scheduler.scheduler.state_dict()["last_epoch"],
        )
        self.assertAlmostEqual(
            restored_optimizer.param_groups[0]["lr"],
            expected_optimizer.param_groups[0]["lr"],
            places=12,
        )

    def test_training_entrypoint_saves_full_resume_payload_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_tiny_processed_dataset(tmp_dir, "tiny_resume_train")
            checkpoints_dir = os.path.join(tmp_dir, "checkpoints")
            argv = [
                "train_imitation_learning_pyg",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "tiny_resume_train",
                "--save_termination_state",
                "--obs_radius",
                "5",
                "--num_samples",
                "2",
                "--num_epochs",
                "1",
                "--validation_every_epochs",
                "1",
                "--batch_size",
                "1",
                "--max_train_batches",
                "1",
                "--skip_validation",
                "--checkpoints_dir",
                checkpoints_dir,
            ]

            with mock.patch("sys.argv", argv):
                train_imitation_learning_pyg.main()

            epoch_checkpoint = os.path.join(checkpoints_dir, "epoch_0.pt")
            last_checkpoint = os.path.join(checkpoints_dir, "last.pt")
            self.assertTrue(os.path.exists(epoch_checkpoint))
            self.assertTrue(os.path.exists(last_checkpoint))

            payload = _load_resume_training_checkpoint(
                last_checkpoint, map_location="cpu"
            )
            self.assertEqual(payload["checkpoint_type"], TRAINING_CHECKPOINT_TYPE)
            self.assertEqual(payload["format_version"], TRAINING_CHECKPOINT_VERSION)
            self.assertEqual(payload["epoch"], 0)
            self.assertIn("optimizer_state_dict", payload)
            self.assertIn("lr_scheduler_state_dict", payload)
            self.assertIn("rng_state", payload)
            self.assertIn("dataset_state", payload)
            self.assertIn("training_state", payload)
            self.assertEqual(payload["args"]["checkpoints_dir"], checkpoints_dir)
            self.assertIsNone(payload["args"]["resume_checkpoint_path"])
            self.assertEqual(payload["dataset_state"]["mode"], "unsharded")
            self.assertEqual(payload["dataset_state"]["train_dataset_len"], 1)
            self.assertEqual(payload["dataset_state"]["validation_dataset_len"], 0)

    def test_training_entrypoint_saves_progress_before_validation_rollout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_tiny_processed_dataset(
                tmp_dir,
                "tiny_validation_checkpoint_timing",
                num_maps=4,
            )
            checkpoints_dir = os.path.join(tmp_dir, "checkpoints")
            sentinel_message = "validation_started_after_checkpoint_save"
            original_progress_logger = train_imitation_learning_pyg.ProgressLogger
            test_case = self

            class CheckpointAwareProgressLogger(original_progress_logger):
                def __init__(self, label, total, **kwargs):
                    if label.startswith("Validation accuracy epoch"):
                        self_assert_epoch_checkpoint = os.path.join(
                            checkpoints_dir, "epoch_0.pt"
                        )
                        self_assert_last_checkpoint = os.path.join(
                            checkpoints_dir, "last.pt"
                        )
                        test_case.assertTrue(
                            os.path.exists(self_assert_epoch_checkpoint)
                        )
                        test_case.assertTrue(
                            os.path.exists(self_assert_last_checkpoint)
                        )
                        raise RuntimeError(sentinel_message)
                    super().__init__(label, total, **kwargs)

            argv = [
                "train_imitation_learning_pyg",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "tiny_validation_checkpoint_timing",
                "--save_termination_state",
                "--obs_radius",
                "5",
                "--num_samples",
                "4",
                "--num_epochs",
                "1",
                "--validation_every_epochs",
                "1",
                "--validation_fraction",
                "0.25",
                "--test_fraction",
                "0.25",
                "--initial_val_size",
                "1",
                "--batch_size",
                "1",
                "--max_train_batches",
                "1",
                "--checkpoints_dir",
                checkpoints_dir,
            ]

            with self.assertRaisesRegex(RuntimeError, sentinel_message):
                with mock.patch.object(
                    train_imitation_learning_pyg,
                    "ProgressLogger",
                    new=CheckpointAwareProgressLogger,
                ):
                    with mock.patch("sys.argv", argv):
                        train_imitation_learning_pyg.main()

    def test_training_entrypoint_sharded_sequence_emits_runtime_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_tiny_sharded_processed_dataset(tmp_dir, "tiny_sharded_sequence")
            checkpoints_dir = os.path.join(tmp_dir, "checkpoints")
            tensorboard_dir = os.path.join(tmp_dir, "runs")
            argv = [
                "train_imitation_learning_pyg",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "tiny_sharded_sequence",
                "--save_termination_state",
                "--obs_radius",
                "5",
                "--num_samples",
                "2",
                "--dataset_seed",
                "42",
                "--obstacle_density_max",
                "0.7",
                "--ensure_grid_config_is_generatable",
                "--use_shards",
                "--use_lists",
                "--sequence_training",
                "--coordination_state_size",
                "4",
                "--num_epochs",
                "1",
                "--validation_every_epochs",
                "1",
                "--batch_size",
                "1",
                "--max_train_batches",
                "1",
                "--skip_validation",
                "--validation_fraction",
                "0.0",
                "--test_fraction",
                "0.0",
                "--dataloader_num_workers",
                "0",
                "--tensorboard_dir",
                tensorboard_dir,
                "--checkpoints_dir",
                checkpoints_dir,
            ]

            with mock.patch("sys.argv", argv):
                train_imitation_learning_pyg.main()

            accumulator = event_accumulator.EventAccumulator(tensorboard_dir)
            accumulator.Reload()
            scalar_tags = set(accumulator.Tags()["scalars"])
            for tag in (
                "train_batch_loss",
                "train_batch_accuracy",
                "train_sequence_first_batch_latency_sec",
                "train_sequence_batch_wait_sec",
                "train_sequence_state_restore_sec",
                "train_sequence_data_to_device_sec",
                "train_sequence_forward_sec",
                "train_sequence_state_stash_sec",
                "train_sequence_on_step_sec",
                "train_sequence_loss_compute_sec",
                "train_sequence_backward_sec",
                "train_sequence_optimizer_step_sec",
                "train_sequence_mean_batch_compute_sec",
                "train_sequence_shard_loads",
                "train_sequence_processed_payload_load_sec",
                "train_sequence_shard_dataset_materialization_sec",
                "train_sequence_collate_time_sec",
            ):
                self.assertIn(tag, scalar_tags)
            batch_loss_events = accumulator.Scalars("train_batch_loss")
            self.assertEqual(len(batch_loss_events), 1)
            self.assertEqual(batch_loss_events[0].step, 0)
            self.assertGreater(batch_loss_events[0].value, 0.0)
            batch_accuracy_events = accumulator.Scalars("train_batch_accuracy")
            self.assertEqual(len(batch_accuracy_events), 1)
            self.assertEqual(batch_accuracy_events[0].step, 0)
            self.assertGreaterEqual(batch_accuracy_events[0].value, 0.0)
            self.assertLessEqual(batch_accuracy_events[0].value, 1.0)
            metrics_csv_path = _default_batch_metrics_csv_path(
                SimpleNamespace(dataset_dir=tmp_dir, checkpoints_dir=checkpoints_dir)
            )
            self.assertTrue(os.path.exists(metrics_csv_path))
            with open(metrics_csv_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["optimization_step"]), 0)
            self.assertEqual(int(rows[0]["epoch"]), 0)
            self.assertEqual(int(rows[0]["batch_idx"]), 1)
            self.assertGreater(float(rows[0]["train_batch_loss"]), 0.0)
            self.assertGreaterEqual(float(rows[0]["train_batch_accuracy"]), 0.0)
            self.assertLessEqual(float(rows[0]["train_batch_accuracy"]), 1.0)

    def test_training_entrypoint_writes_validation_metrics_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_tiny_sharded_processed_dataset(tmp_dir, "tiny_validation_metrics")
            checkpoints_dir = os.path.join(tmp_dir, "checkpoints")
            tensorboard_dir = os.path.join(tmp_dir, "runs")
            argv = [
                "train_imitation_learning_pyg",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "tiny_validation_metrics",
                "--save_termination_state",
                "--obs_radius",
                "5",
                "--num_samples",
                "2",
                "--dataset_seed",
                "42",
                "--obstacle_density_max",
                "0.7",
                "--ensure_grid_config_is_generatable",
                "--use_shards",
                "--use_lists",
                "--sequence_training",
                "--coordination_state_size",
                "4",
                "--num_epochs",
                "1",
                "--validation_every_epochs",
                "1",
                "--batch_size",
                "1",
                "--max_train_batches",
                "1",
                "--validate_sequence_training_dataset",
                "--validation_fraction",
                "0.5",
                "--test_fraction",
                "0.0",
                "--dataloader_num_workers",
                "0",
                "--tensorboard_dir",
                tensorboard_dir,
                "--checkpoints_dir",
                checkpoints_dir,
            ]

            with mock.patch("sys.argv", argv):
                train_imitation_learning_pyg.main()

            validation_csv_path = _default_validation_metrics_csv_path(
                SimpleNamespace(dataset_dir=tmp_dir, checkpoints_dir=checkpoints_dir)
            )
            self.assertTrue(os.path.exists(validation_csv_path))
            with open(validation_csv_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["epoch"]), 0)
            self.assertNotEqual(rows[0]["validation_accuracy"], "")
            self.assertNotEqual(rows[0]["validation_success_rate"], "")
            self.assertNotEqual(rows[0]["validation_average_makespan"], "")
            self.assertNotEqual(rows[0]["validation_average_partial_success_rate"], "")
            self.assertNotEqual(rows[0]["validation_average_sum_of_costs"], "")

    def test_training_entrypoint_resume_from_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_tiny_processed_dataset(tmp_dir, "tiny_resume_roundtrip")
            initial_checkpoints_dir = os.path.join(tmp_dir, "checkpoints_initial")
            original_save_training_checkpoint = (
                train_imitation_learning_pyg._save_training_checkpoint
            )

            def stop_after_first_last_checkpoint(checkpoint_path, **kwargs):
                original_save_training_checkpoint(checkpoint_path, **kwargs)
                checkpoint_name = os.path.basename(str(checkpoint_path))
                if checkpoint_name == "last.pt" and kwargs["epoch"] == 0:
                    raise RuntimeError("stop after checkpoint")

            initial_argv = [
                "train_imitation_learning_pyg",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "tiny_resume_roundtrip",
                "--save_termination_state",
                "--obs_radius",
                "5",
                "--num_samples",
                "2",
                "--num_epochs",
                "2",
                "--batch_size",
                "1",
                "--max_train_batches",
                "1",
                "--skip_validation",
                "--checkpoints_dir",
                initial_checkpoints_dir,
            ]
            with mock.patch.object(
                train_imitation_learning_pyg,
                "_save_training_checkpoint",
                new=stop_after_first_last_checkpoint,
            ):
                with mock.patch("sys.argv", initial_argv):
                    with self.assertRaisesRegex(RuntimeError, "stop after checkpoint"):
                        train_imitation_learning_pyg.main()

            resume_source = os.path.join(initial_checkpoints_dir, "last.pt")
            resumed_checkpoints_dir = os.path.join(tmp_dir, "checkpoints_resumed")
            resumed_argv = [
                "train_imitation_learning_pyg",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "tiny_resume_roundtrip",
                "--save_termination_state",
                "--obs_radius",
                "5",
                "--num_samples",
                "2",
                "--num_epochs",
                "2",
                "--batch_size",
                "1",
                "--max_train_batches",
                "1",
                "--skip_validation",
                "--resume_checkpoint_path",
                resume_source,
                "--checkpoints_dir",
                resumed_checkpoints_dir,
            ]
            with mock.patch("sys.argv", resumed_argv):
                train_imitation_learning_pyg.main()

            resumed_last = os.path.join(resumed_checkpoints_dir, "last.pt")
            self.assertTrue(os.path.exists(resumed_last))

            initial_payload = _load_resume_training_checkpoint(
                resume_source, map_location="cpu"
            )
            resumed_payload = _load_resume_training_checkpoint(
                resumed_last, map_location="cpu"
            )
            self.assertEqual(resumed_payload["epoch"], 1)
            self.assertEqual(
                resumed_payload["training_state"]["best_validation_accuracy"],
                initial_payload["training_state"]["best_validation_accuracy"],
            )
            self.assertEqual(
                resumed_payload["training_state"]["best_validation_success_rate"],
                initial_payload["training_state"]["best_validation_success_rate"],
            )
            self.assertEqual(
                resumed_payload["dataset_state"], initial_payload["dataset_state"]
            )
            self.assertEqual(
                resumed_payload["args"]["resume_checkpoint_path"], resume_source
            )
            self.assertEqual(
                resumed_payload["args"]["checkpoints_dir"], resumed_checkpoints_dir
            )

    def test_training_entrypoint_resume_from_completed_last_checkpoint_with_extended_num_epochs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_tiny_processed_dataset(tmp_dir, "tiny_resume_extended")
            initial_checkpoints_dir = os.path.join(tmp_dir, "checkpoints_initial")
            initial_argv = [
                "train_imitation_learning_pyg",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "tiny_resume_extended",
                "--save_termination_state",
                "--obs_radius",
                "5",
                "--num_samples",
                "2",
                "--num_epochs",
                "2",
                "--batch_size",
                "1",
                "--max_train_batches",
                "1",
                "--skip_validation",
                "--checkpoints_dir",
                initial_checkpoints_dir,
            ]
            with mock.patch("sys.argv", initial_argv):
                train_imitation_learning_pyg.main()

            resume_source = os.path.join(initial_checkpoints_dir, "last.pt")
            resumed_checkpoints_dir = os.path.join(tmp_dir, "checkpoints_resumed")
            resumed_argv = [
                "train_imitation_learning_pyg",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "tiny_resume_extended",
                "--save_termination_state",
                "--obs_radius",
                "5",
                "--num_samples",
                "2",
                "--num_epochs",
                "4",
                "--batch_size",
                "1",
                "--max_train_batches",
                "1",
                "--skip_validation",
                "--resume_checkpoint_path",
                resume_source,
                "--checkpoints_dir",
                resumed_checkpoints_dir,
            ]
            with mock.patch("sys.argv", resumed_argv):
                train_imitation_learning_pyg.main()

            resumed_last = os.path.join(resumed_checkpoints_dir, "last.pt")
            self.assertTrue(os.path.exists(resumed_last))

            resumed_payload = _load_resume_training_checkpoint(
                resumed_last, map_location="cpu"
            )
            self.assertEqual(resumed_payload["epoch"], 3)
            self.assertEqual(resumed_payload["args"]["num_epochs"], 4)

    def test_trained_cs_checkpoint_keeps_distinct_states_on_two_agent_map(self):
        checkpoint_path = os.path.join(
            "checkpoints",
            "hmagat_cs_sequence_32_pilot_800b_1ep_bs20_gpu_resnet_residuals_all_widened",
            "epoch_0.pt",
        )
        if not os.path.exists(checkpoint_path):
            self.skipTest(
                "Expected local HMAGAT-CS pilot checkpoint for two-agent state test."
            )

        args = _checkpoint_runtime_args()
        model, hypergraph_model, dataset_kwargs = get_model(args, torch.device("cpu"))
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        grid_config = GridConfig(
            map=[
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            agents_xy=[[1, 2], [5, 1]],
            targets_xy=[[1, 5], [4, 4]],
            obs_radius=args.obs_radius,
            collision_system="soft",
            on_target="nothing",
            observation_type="MAPF",
            max_episode_steps=32,
            seed=123,
        )
        env = pogema_v0(grid_config=grid_config)
        observations, _infos = env.reset()
        runtime_data_generator = get_runtime_data_generator(
            grid_config=grid_config,
            args=args,
            hypergraph_model=hypergraph_model,
            dataset_kwargs=dataset_kwargs,
            use_target_vec="target-vec",
        )
        data = runtime_data_generator(observations, env)

        model.in_simulation(True)
        with torch.no_grad():
            out = model(data.x, data)

        state = model._coordination_state
        self.assertEqual(tuple(out.shape), (2, 5))
        self.assertEqual(tuple(state.shape), (2, args.coordination_state_size))
        self.assertGreater(torch.norm(state[0] - state[1]).item(), 1e-6)

    def test_dataset_legacy_fallback_is_logged_as_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = os.path.join(tmp_dir, "processed_dataset")
            os.makedirs(dataset_dir)
            dataset_path = os.path.join(dataset_dir, "present.pkl")
            expected_dataset = {"ok": True}
            with open(dataset_path, "wb") as f:
                pickle.dump(expected_dataset, f)

            args = SimpleNamespace(dataset_dir=tmp_dir)

            dataset, warnings = _capture_loguru_warnings(
                lambda: load_dataset(
                    [
                        lambda _args: "missing.pkl",
                        lambda _args: "present.pkl",
                    ],
                    "processed_dataset",
                    args,
                )
            )

        self.assertEqual(dataset, expected_dataset)
        self.assertTrue(
            any("Trying legacy file name fallback" in message for message in warnings)
        )

    def test_dataset_loading_does_not_fallback_on_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = os.path.join(tmp_dir, "processed_dataset")
            os.makedirs(dataset_dir)
            broken_path = os.path.join(dataset_dir, "broken.pkl")
            valid_path = os.path.join(dataset_dir, "valid.pkl")
            with open(broken_path, "wb") as f:
                f.write(b"not-a-pickle")
            with open(valid_path, "wb") as f:
                pickle.dump({"ok": True}, f)

            args = SimpleNamespace(dataset_dir=tmp_dir)

            _, warnings = _capture_loguru_warnings(
                lambda: self.assertRaises(
                    pickle.UnpicklingError,
                    load_dataset,
                    [
                        lambda _args: "broken.pkl",
                        lambda _args: "valid.pkl",
                    ],
                    "processed_dataset",
                    args,
                )
            )

        self.assertFalse(
            any("Trying legacy file name fallback" in message for message in warnings)
        )

    def test_validation_schedule_rejects_non_positive_frequency(self):
        _, warnings = _capture_loguru_warnings(
            lambda: self.assertRaisesRegex(
                ValueError,
                "validation_every_epochs must be positive",
                validate_training_args_contract,
                SimpleNamespace(validation_every_epochs=0),
            )
        )

        self.assertTrue(
            any("validation_every_epochs must be positive" in message for message in warnings)
        )

    def test_sharded_dataset_cache_size_rejects_non_positive_value(self):
        _, warnings = _capture_loguru_warnings(
            lambda: self.assertRaisesRegex(
                ValueError,
                "sharded_dataset_cache_size must be positive",
                validate_training_args_contract,
                SimpleNamespace(
                    validation_every_epochs=1,
                    sharded_dataset_cache_size=0,
                ),
            )
        )

        self.assertTrue(
            any(
                "sharded_dataset_cache_size must be positive" in message
                for message in warnings
            )
        )

    def test_generate_additional_data_non_sharded_plain_dataset_raises_value_error_not_attribute_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = os.path.join(tmp_dir, "raw_expert_predictions")
            os.makedirs(raw_dir, exist_ok=True)
            dataset_path = os.path.join(raw_dir, "plain.pkl")
            with open(dataset_path, "wb") as f:
                pickle.dump([("dummy_obs", "dummy_actions", "dummy_term")], f)

            argv = [
                "generate_additional_data",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "plain",
                "--num_samples",
                "1",
                "--dataset_seed",
                "42",
            ]

            def run():
                with mock.patch("sys.argv", argv):
                    generate_additional_data.main()

            _, warnings = _capture_loguru_warnings(
                lambda: self.assertRaisesRegex(
                    ValueError,
                    "Dataset is expected to have a seed_mask",
                    run,
                )
            )

        self.assertFalse(any("AttributeError" in message for message in warnings))

    def test_hypergraph_dataset_exposes_first_step_for_training_loop(self):
        dense_dataset = (
            [
                torch.randn(4, 3, 13, 13),
                torch.randn(4, 3, 13, 13),
                torch.randn(4, 3, 13, 13),
            ],
            [torch.ones(4, 4), torch.ones(4, 4), torch.ones(4, 4)],
            [torch.zeros(4, dtype=torch.long)] * 3,
            [torch.zeros(4, dtype=torch.bool)] * 3,
            torch.tensor([0, 0, 1]),
        )
        hyperedge_indices = (
            [
                torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
                torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
                torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
            ],
            [
                torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long),
                torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long),
                torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long),
            ],
        )
        dataset = MAPFHypergraphDataset(
            dense_dataset,
            hyperedge_indices,
            use_edge_attr=False,
        )

        self.assertTrue(dataset[0].first_step.item())
        self.assertFalse(dataset[1].first_step.item())
        self.assertTrue(dataset[2].first_step.item())

    def test_graph_and_hypergraph_datasets_expose_agent_id_for_sequence_checks(self):
        dense_dataset = (
            [torch.randn(4, 3, 13, 13), torch.randn(4, 3, 13, 13)],
            [torch.ones(4, 4), torch.ones(4, 4)],
            [torch.zeros(4, dtype=torch.long)] * 2,
            [torch.zeros(4, dtype=torch.bool)] * 2,
            torch.tensor([0, 0]),
        )
        graph_dataset = MAPFGraphDataset(dense_dataset, use_edge_attr=False)
        hypergraph_dataset = self._small_hypergraph_snapshot_dataset(
            graph_map_id=torch.tensor([0, 0])
        )

        self.assertEqual(graph_dataset[0].agent_id.tolist(), [0, 1, 2, 3])
        self.assertEqual(hypergraph_dataset[0].agent_id.tolist(), [0, 1, 2, 3])
        self.assertTrue(
            validate_sequence_dataset_assumptions(graph_dataset)["checked_agent_id"]
        )
        self.assertTrue(
            validate_sequence_dataset_assumptions(hypergraph_dataset)[
                "checked_agent_id"
            ]
        )

    def test_sequence_grouping_preserves_contiguous_episode_order(self):
        groups = group_indices_by_graph_map_id(torch.tensor([0, 0, 0, 1, 1, 2]))

        self.assertEqual(groups, [[0, 1, 2], [3, 4], [5]])

    def test_sequence_grouping_rejects_non_contiguous_episode_ids(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            group_indices_by_graph_map_id(torch.tensor([0, 1, 0]))

    def test_sequence_assumption_validation_accepts_stable_agent_order(self):
        snapshots = [
            _graph_step(_observations(), first_step=True),
            _graph_step(_observations(), first_step=False),
            _graph_step(_observations(), first_step=True),
        ]
        for snapshot in snapshots:
            snapshot.agent_id = torch.arange(4)
        snapshot_dataset = _SnapshotList(
            snapshots,
            graph_map_id=torch.tensor([0, 0, 1]),
        )

        summary = validate_sequence_dataset_assumptions(snapshot_dataset)

        self.assertEqual(summary["num_episodes"], 2)
        self.assertEqual(summary["num_snapshots"], 3)
        self.assertEqual(summary["episode_lengths"], [2, 1])
        self.assertTrue(summary["checked_agent_id"])

    def test_sequence_assumption_validation_rejects_agent_order_change(self):
        snapshots = [
            _graph_step(_observations(), first_step=True),
            _graph_step(_observations(), first_step=False),
        ]
        snapshots[0].agent_id = torch.tensor([0, 1, 2, 3])
        snapshots[1].agent_id = torch.tensor([0, 2, 1, 3])
        snapshot_dataset = _SnapshotList(
            snapshots,
            graph_map_id=torch.tensor([0, 0]),
        )

        with self.assertRaisesRegex(ValueError, "agent_id order changed"):
            validate_sequence_dataset_assumptions(snapshot_dataset)

    def test_sequence_assumption_validation_warns_without_agent_ids(self):
        snapshot_dataset = _SnapshotList(
            [
                _graph_step(_observations(), first_step=True),
                _graph_step(_observations(), first_step=False),
            ],
            graph_map_id=torch.tensor([0, 0]),
        )

        summary, warnings = _capture_loguru_warnings(
            lambda: validate_sequence_dataset_assumptions(snapshot_dataset)
        )

        self.assertFalse(summary["checked_agent_id"])
        self.assertTrue(
            any("cannot prove stable agent row order" in message for message in warnings)
        )

    def test_sequence_assumption_validation_rejects_bad_first_step(self):
        snapshot_dataset = _SnapshotList(
            [
                _graph_step(_observations(), first_step=True),
                _graph_step(_observations(), first_step=True),
            ],
            graph_map_id=torch.tensor([0, 0]),
        )

        with self.assertRaisesRegex(ValueError, "first_step"):
            validate_sequence_dataset_assumptions(snapshot_dataset)

    def test_dense_sequence_audit_accepts_row_order_contract(self):
        dense_dataset = (
            [
                torch.randn(4, 3, 13, 13),
                torch.randn(4, 3, 13, 13),
                torch.randn(6, 3, 13, 13),
            ],
            [torch.ones(4, 4), torch.ones(4, 4), torch.ones(6, 6)],
            [torch.zeros(4, dtype=torch.long)] * 2
            + [torch.zeros(6, dtype=torch.long)],
            [torch.zeros(4, dtype=torch.bool)] * 2
            + [torch.zeros(6, dtype=torch.bool)],
            torch.tensor([0, 0, 1]),
        )

        summary = validate_dense_sequence_dataset_assumptions(dense_dataset)

        self.assertEqual(summary["num_episodes"], 2)
        self.assertEqual(summary["num_snapshots"], 3)
        self.assertEqual(summary["episode_lengths"], [2, 1])
        self.assertTrue(summary["checked_agent_id"])

    def test_dense_sequence_audit_rejects_agent_count_change_inside_episode(self):
        dense_dataset = (
            [torch.randn(4, 3, 13, 13), torch.randn(5, 3, 13, 13)],
            [torch.ones(4, 4), torch.ones(5, 5)],
            [torch.zeros(4, dtype=torch.long), torch.zeros(5, dtype=torch.long)],
            [torch.zeros(4, dtype=torch.bool), torch.zeros(5, dtype=torch.bool)],
            torch.tensor([0, 0]),
        )

        with self.assertRaisesRegex(ValueError, "Agent row count changed"):
            validate_dense_sequence_dataset_assumptions(dense_dataset)

    def test_lightweight_audit_skips_hypergraphs_and_additional_data_loads(self):
        dense_dataset = (
            [torch.randn(4, 3, 13, 13), torch.randn(4, 3, 13, 13)],
            [torch.ones(4, 4), torch.ones(4, 4)],
            [torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)],
            [torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)],
            torch.tensor([0, 0]),
        )
        loaded_dirs = []

        def fake_load_dataset(_funcs, dir_name, _args):
            loaded_dirs.append(dir_name)
            if dir_name != "processed_dataset":
                raise AssertionError(f"Unexpected lightweight audit load: {dir_name}")
            return dense_dataset

        argv = [
            "audit_sequence_dataset",
            "--dataset_dir",
            "/unused",
            "--override_name",
            "unused",
            "--imitation_learning_model",
            "DirectionalHMAGAT",
            "--add_data_cost_to_go",
        ]
        with mock.patch.object(audit_sequence_dataset, "load_dataset", fake_load_dataset):
            with mock.patch("sys.argv", argv):
                audit_sequence_dataset.main()

        self.assertEqual(loaded_dirs, ["processed_dataset"])

    def test_convert_list_path_reuses_numpy_storage(self):
        array = np.array([1, 2, 3], dtype=np.int64)

        tensor = _as_torch_tensor(array)
        tensor[0] = 9

        self.assertEqual(array[0], 9)

    def test_convert_releases_raw_dataset_before_pickle_dump(self):
        class WeakList(list):
            pass

        dataset_ref = None

        def fake_load_dataset(_funcs, _dir_name, _args):
            nonlocal dataset_ref
            dataset = WeakList([("raw",)])
            dataset_ref = weakref.ref(dataset)
            return dataset

        def fake_generate_graph_dataset(dataset, *_args, **_kwargs):
            self.assertIs(dataset_ref(), dataset)
            return ("processed",)

        def fake_pickle_dump(_graph_dataset, _file):
            self.assertIsNone(dataset_ref())

        argv = [
            "convert_to_imitation_dataset",
            "--dataset_dir",
            tempfile.mkdtemp(),
            "--override_name",
            "unused",
            "--save_termination_state",
        ]
        with mock.patch.object(
            convert_to_imitation_dataset,
            "load_dataset",
            fake_load_dataset,
        ):
            with mock.patch.object(
                convert_to_imitation_dataset,
                "generate_graph_dataset",
                fake_generate_graph_dataset,
            ):
                with mock.patch.object(
                    convert_to_imitation_dataset.pickle,
                    "dump",
                    fake_pickle_dump,
                ):
                    with mock.patch("sys.argv", argv):
                        convert_to_imitation_dataset.main()

    def test_sequence_dataset_wraps_existing_snapshot_dataset(self):
        snapshot_dataset = self._small_hypergraph_snapshot_dataset()
        sequence_dataset = MAPFSequenceDataset(snapshot_dataset)

        self.assertEqual(len(sequence_dataset), 2)
        self.assertEqual(len(sequence_dataset[0]), 2)
        self.assertEqual(len(sequence_dataset[1]), 1)
        self.assertTrue(sequence_dataset[0][0].first_step.item())
        self.assertFalse(sequence_dataset[0][1].first_step.item())
        self.assertTrue(sequence_dataset[1][0].first_step.item())

    def test_sequence_collate_returns_timestep_batches_and_active_episode_indices(self):
        snapshot_dataset = self._small_hypergraph_snapshot_dataset()
        sequence_dataset = MAPFSequenceDataset(snapshot_dataset)

        sequence_batch = collate_mapf_sequences(
            [sequence_dataset[0], sequence_dataset[1]]
        )

        self.assertEqual(len(sequence_batch), 2)
        self.assertEqual(sequence_batch.sequence_lengths.tolist(), [2, 1])
        self.assertEqual(sequence_batch.active_episode_indices[0].tolist(), [0, 1])
        self.assertEqual(sequence_batch.active_episode_indices[1].tolist(), [0])
        self.assertEqual(sequence_batch.active_episode_index_lists[0], (0, 1))
        self.assertEqual(sequence_batch.active_episode_index_lists[1], (0,))
        self.assertEqual(sequence_batch.timesteps[0].num_graphs, 2)
        self.assertEqual(sequence_batch.timesteps[1].num_graphs, 1)
        self.assertEqual(sequence_batch.graph_counts, (2, 1))
        self.assertEqual(sequence_batch.first_step_graph_counts, (2, 0))
        self.assertEqual(sequence_batch.node_row_counts, (8, 4))
        self.assertEqual(sequence_batch.timestep_ptr_lists[0], (0, 4, 8))
        self.assertEqual(sequence_batch.timestep_ptr_lists[1], (0, 4))
        self.assertEqual(sequence_batch.timesteps[0].first_step.tolist(), [True, True])
        self.assertEqual(sequence_batch.timesteps[1].first_step.tolist(), [False])

    def test_sequence_collate_matches_reference_behavior_on_variable_lengths(self):
        sequences = [
            [
                _graph_step(_observations(), True),
                _graph_step(_observations(), False),
                _graph_step(_observations(), False),
            ],
            [_graph_step(_observations(), True)],
            [_graph_step(_observations(), True), _graph_step(_observations(), False)],
        ]

        actual = collate_mapf_sequences(sequences)
        expected = _reference_collate_mapf_sequences(sequences)

        self.assertEqual(actual.sequence_lengths.tolist(), expected.sequence_lengths.tolist())
        for actual_indices, expected_indices in zip(
            actual.active_episode_indices, expected.active_episode_indices
        ):
            self.assertEqual(actual_indices.tolist(), expected_indices.tolist())
        for actual_batch, expected_batch in zip(actual.timesteps, expected.timesteps):
            self.assertEqual(actual_batch.num_graphs, expected_batch.num_graphs)
            torch.testing.assert_close(actual_batch.x, expected_batch.x)
            torch.testing.assert_close(actual_batch.y, expected_batch.y)
            torch.testing.assert_close(actual_batch.ptr, expected_batch.ptr)
            self.assertEqual(
                actual_batch.first_step.tolist(), expected_batch.first_step.tolist()
            )

    def test_sequence_collate_accepts_lightweight_sequence_append_protocol(self):
        sequences = [
            _LightweightSequence(
                [
                    _graph_step(_observations(), True),
                    _graph_step(_observations(), False),
                ]
            ),
            _LightweightSequence([_graph_step(_observations(), True)]),
        ]

        actual = collate_mapf_sequences(sequences)
        expected = _reference_collate_mapf_sequences(
            [[sequence[idx] for idx in range(len(sequence))] for sequence in sequences]
        )

        self.assertEqual(actual.sequence_lengths.tolist(), expected.sequence_lengths.tolist())
        self.assertEqual(actual.graph_counts, (2, 1))
        self.assertEqual(actual.first_step_graph_counts, (2, 0))
        self.assertEqual(actual.node_row_counts, (8, 4))
        for actual_indices, expected_indices in zip(
            actual.active_episode_indices, expected.active_episode_indices
        ):
            self.assertEqual(actual_indices.tolist(), expected_indices.tolist())
        for actual_batch, expected_batch in zip(actual.timesteps, expected.timesteps):
            torch.testing.assert_close(actual_batch.x, expected_batch.x)
            torch.testing.assert_close(actual_batch.ptr, expected_batch.ptr)

    def test_sequence_batch_collator_tracks_collate_metrics(self):
        snapshot_dataset = self._small_hypergraph_snapshot_dataset()
        sequence_dataset = MAPFSequenceDataset(snapshot_dataset)
        collator = SequenceBatchCollator()

        batch = collator([sequence_dataset[0], sequence_dataset[1]])
        metrics = collator.metrics_snapshot()

        self.assertIsInstance(batch, MAPFSequenceBatch)
        self.assertEqual(metrics["collate_calls"], 1)
        self.assertGreaterEqual(metrics["collate_time_sec"], 0.0)
        self.assertGreaterEqual(metrics["last_collate_time_sec"], 0.0)

    def test_sequence_step_runtime_tracker_and_epoch_metrics_emit_mean_compute(self):
        step_tracker = SequenceStepRuntimeTracker()
        step_tracker.record_timestep()
        step_tracker.record_timestep()
        step_tracker.record_state_restore(0.05)
        step_tracker.record_data_to_device(0.2)
        step_tracker.record_forward(0.4)
        step_tracker.record_state_stash(0.07)
        step_tracker.record_on_step(0.03)
        step_tracker.record_loss_compute(0.1)

        epoch_tracker = EpochRuntimeTracker()
        epoch_tracker.record_batch_wait(0.3)
        epoch_tracker.record_backward(0.5)
        epoch_tracker.record_optimizer_step(0.1)
        epoch_tracker.record_batch()
        epoch_tracker.record_batch()

        metrics = _sequence_runtime_metrics(
            prefix="train_sequence",
            step_runtime=step_tracker,
            epoch_runtime=epoch_tracker,
            first_batch_latency_sec=0.25,
        )

        self.assertEqual(metrics["train_sequence_timesteps"], 2.0)
        self.assertAlmostEqual(metrics["train_sequence_state_restore_sec"], 0.05)
        self.assertAlmostEqual(metrics["train_sequence_data_to_device_sec"], 0.2)
        self.assertAlmostEqual(metrics["train_sequence_forward_sec"], 0.4)
        self.assertAlmostEqual(metrics["train_sequence_state_stash_sec"], 0.07)
        self.assertAlmostEqual(metrics["train_sequence_on_step_sec"], 0.03)
        self.assertAlmostEqual(metrics["train_sequence_loss_compute_sec"], 0.1)
        self.assertAlmostEqual(metrics["train_sequence_batch_wait_sec"], 0.3)
        self.assertAlmostEqual(metrics["train_sequence_backward_sec"], 0.5)
        self.assertAlmostEqual(metrics["train_sequence_optimizer_step_sec"], 0.1)
        self.assertEqual(metrics["train_sequence_num_batches"], 2.0)
        self.assertAlmostEqual(metrics["train_sequence_first_batch_latency_sec"], 0.25)
        self.assertAlmostEqual(
            metrics["train_sequence_mean_batch_compute_sec"],
            (0.05 + 0.2 + 0.4 + 0.07 + 0.03 + 0.1 + 0.5 + 0.1) / 2,
        )

    def test_sequence_collate_rejects_empty_batches(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            collate_mapf_sequences([])

    def test_compute_sequence_loss_runs_backward_over_timestep_batches(self):
        snapshot_dataset = self._small_hypergraph_snapshot_dataset(
            graph_map_id=torch.tensor([0, 0, 1, 1])
        )
        sequence_dataset = MAPFSequenceDataset(snapshot_dataset)
        sequence_batch = collate_mapf_sequences(
            [sequence_dataset[0], sequence_dataset[1]]
        )
        model = _model(
            coordination_state_size=4,
            gnn_type="DirectionalHMAGAT",
            gnn_kwargs={
                "hyperedge_feature_generator": "magat",
                "final_feature_generator": "magat",
            },
        )
        model.train()
        seen_timesteps = []

        loss = compute_sequence_loss(
            model,
            sequence_batch,
            _SequenceLoss(),
            on_step=lambda out, data: seen_timesteps.append(
                (tuple(out.shape), data.num_graphs)
            ),
        )
        loss.backward()

        self.assertEqual(tuple(loss.shape), ())
        self.assertEqual(seen_timesteps, [((8, 5), 2), ((8, 5), 2)])
        self.assertTrue(model.detach_coordination_state)
        self.assertIsNotNone(model.coordination_state_cell.weight_ih.grad)
        self.assertGreater(
            torch.norm(model.coordination_state_cell.weight_ih.grad).item(),
            0.0,
        )

    def test_compute_sequence_loss_runs_under_cuda_bfloat16_autocast(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for AMP autocast test.")
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA bfloat16 support is required for AMP autocast test.")

        snapshot_dataset = self._small_hypergraph_snapshot_dataset(
            graph_map_id=torch.tensor([0, 0, 1, 1])
        )
        sequence_dataset = MAPFSequenceDataset(snapshot_dataset)
        sequence_batch = collate_mapf_sequences(
            [sequence_dataset[0], sequence_dataset[1]]
        )
        model = _model(
            coordination_state_size=4,
            gnn_type="DirectionalHMAGAT",
            gnn_kwargs={
                "hyperedge_feature_generator": "magat",
                "final_feature_generator": "magat",
            },
        ).to("cuda")
        model.train()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compute_sequence_loss(
                model,
                sequence_batch,
                _SequenceLoss(),
                device=torch.device("cuda"),
            )
        loss.backward()

        self.assertEqual(tuple(loss.shape), ())
        self.assertEqual(loss.device.type, "cuda")
        self.assertIsNotNone(model.coordination_state_cell.weight_ih.grad)
        self.assertGreater(
            torch.norm(model.coordination_state_cell.weight_ih.grad).item(),
            0.0,
        )

    def test_sequence_loss_reorders_state_for_variable_length_episode_batch(self):
        obs_ep0_t0 = _observations()
        obs_ep1_t0 = _observations().requires_grad_(True)
        obs_ep1_t1 = _observations()
        sequence_batch = collate_mapf_sequences(
            [
                [_graph_step(obs_ep0_t0, first_step=True)],
                [
                    _graph_step(obs_ep1_t0, first_step=True),
                    _graph_step(obs_ep1_t1, first_step=False),
                ],
            ]
        )
        model = _model(coordination_state_size=4)
        model.train()

        loss = compute_sequence_loss(model, sequence_batch, _SecondTimestepLoss())
        loss.backward()

        self.assertEqual(sequence_batch.active_episode_indices[1].tolist(), [1])
        self.assertIsNotNone(obs_ep1_t0.grad)
        self.assertGreater(torch.norm(obs_ep1_t0.grad).item(), 0.0)

    def test_sequence_loss_truncated_bptt_detaches_state_between_windows(self):
        obs_0 = _observations().requires_grad_(True)
        obs_1 = _observations()
        sequence_batch = collate_mapf_sequences(
            [
                [
                    _graph_step(obs_0, first_step=True),
                    _graph_step(obs_1, first_step=False),
                ]
            ]
        )
        model = _model(coordination_state_size=4)
        model.train()

        loss = compute_sequence_loss(
            model,
            sequence_batch,
            _SecondTimestepLoss(),
            truncated_bptt_length=1,
        )
        loss.backward()

        self.assertIsNotNone(obs_0.grad)
        self.assertEqual(torch.norm(obs_0.grad).item(), 0.0)

    def test_sequence_first_timestep_without_saved_state_does_not_log_warning(self):
        sequence_batch = collate_mapf_sequences(
            [[_graph_step(_observations(), first_step=True)]]
        )
        model = _model(coordination_state_size=4)
        model.train()

        _, warnings = _capture_loguru_warnings(
            lambda: compute_sequence_loss(model, sequence_batch, _SequenceLoss())
        )

        self.assertFalse(
            any("no stored coordination state" in message for message in warnings)
        )

    def test_sequence_state_fallback_is_logged_as_warning(self):
        sequence_batch = MAPFSequenceBatch(
            timesteps=[
                collate_mapf_sequences([[_graph_step(_observations(), True)]]).timesteps[
                    0
                ],
                collate_mapf_sequences(
                    [
                        [_graph_step(_observations(), False)],
                        [_graph_step(_observations(), False)],
                    ]
                ).timesteps[0],
            ],
            active_episode_indices=[
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0, 1], dtype=torch.long),
            ],
            sequence_lengths=torch.tensor([2, 1]),
        )
        model = _model(coordination_state_size=4)
        model.train()

        _, warnings = _capture_loguru_warnings(
            lambda: compute_sequence_loss(model, sequence_batch, _SequenceLoss())
        )

        self.assertTrue(
            any("no stored coordination state" in message for message in warnings)
        )

    def _small_hypergraph_snapshot_dataset(self, graph_map_id=None):
        if graph_map_id is None:
            graph_map_id = torch.tensor([0, 0, 1])
        num_snapshots = len(graph_map_id)
        dense_dataset = (
            [torch.randn(4, 3, 13, 13) for _ in range(num_snapshots)],
            [torch.ones(4, 4) for _ in range(num_snapshots)],
            [torch.zeros(4, dtype=torch.long) for _ in range(num_snapshots)],
            [torch.zeros(4, dtype=torch.bool) for _ in range(num_snapshots)],
            graph_map_id,
        )
        hyperedge_indices = (
            [
                torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
            ]
            * num_snapshots,
            [
                torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long),
            ]
            * num_snapshots,
        )
        return MAPFHypergraphDataset(
            dense_dataset,
            hyperedge_indices,
            use_edge_attr=False,
        )


if __name__ == "__main__":
    unittest.main()
