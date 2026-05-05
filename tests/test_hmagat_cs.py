import argparse
import os
import pickle
import tempfile
import unittest
import weakref
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch
from loguru import logger
from pogema import GridConfig, pogema_v0
from torch_geometric.data import Data

from hmagat import audit_sequence_dataset
from hmagat import convert_to_imitation_dataset
from hmagat.audit_sequence_dataset import validate_dense_sequence_dataset_assumptions
from hmagat.convert_to_imitation_dataset import _as_torch_tensor
from hmagat.dataset_loading import load_dataset
from hmagat.imitation_dataset_pyg import (
    MAPFGraphDataset,
    MAPFHypergraphDataset,
    MAPFSequenceBatch,
    MAPFSequenceDataset,
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
from hmagat.sequence_training import compute_sequence_loss
from hmagat.training_args import add_training_args
from hmagat.lr_scheduler import get_estimated_total_number_of_steps
from hmagat.train_imitation_learning_pyg import (
    _effective_num_batches,
    _validate_and_warn_batch_limits,
)


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


class HMAGATCSTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_training_args_include_pilot_batch_limits(self):
        parser = argparse.ArgumentParser()
        add_training_args(parser)

        args = parser.parse_args([])
        self.assertIsNone(args.max_train_batches)
        self.assertIsNone(args.max_validation_batches)

        args = parser.parse_args(
            ["--max_train_batches", "7", "--max_validation_batches", "3"]
        )
        self.assertEqual(args.max_train_batches, 7)
        self.assertEqual(args.max_validation_batches, 3)

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
        class _Loader:
            def __len__(self):
                return 100

        args = SimpleNamespace(
            max_train_batches=7,
            num_epochs=3,
            skip_validation=True,
        )

        self.assertEqual(get_estimated_total_number_of_steps(args, _Loader()), 21)

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
        self.assertEqual(sequence_batch.timesteps[0].num_graphs, 2)
        self.assertEqual(sequence_batch.timesteps[1].num_graphs, 1)
        self.assertEqual(sequence_batch.timesteps[0].first_step.tolist(), [True, True])
        self.assertEqual(sequence_batch.timesteps[1].first_step.tolist(), [False])

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

    def test_sequence_state_fallback_is_logged_as_warning(self):
        sequence_batch = MAPFSequenceBatch(
            timesteps=[
                collate_mapf_sequences([[_graph_step(_observations(), True)]]).timesteps[
                    0
                ],
                collate_mapf_sequences([[_graph_step(_observations(), False)]]).timesteps[
                    0
                ],
            ],
            active_episode_indices=[
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1], dtype=torch.long),
            ],
            sequence_lengths=torch.tensor([1, 1]),
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
