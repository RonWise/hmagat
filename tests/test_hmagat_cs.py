import unittest

import torch
from torch_geometric.data import Data

from hmagat.modules.agents import DecentralPlannerGATNet, load_partial_state_dict


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


class HMAGATCSTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

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

    def test_partial_state_loading_skips_new_state_and_whole_resized_decoder_layer(self):
        old_model = _model(coordination_state_size=0)
        new_model = _model(coordination_state_size=4)
        old_state = old_model.state_dict()

        for key, value in old_state.items():
            if torch.is_floating_point(value):
                old_state[key] = torch.full_like(value, 0.25)

        summary = load_partial_state_dict(new_model, old_state, print_prefix="[test]")

        self.assertIn("cnn.convs.0.weight", summary["loaded"])
        self.assertIn("actionsMLP.0.bias", summary["skipped_related"])
        self.assertTrue(
            any(key == "actionsMLP.0.weight" for key, _, _ in summary["skipped_shape"])
        )
        self.assertIn(
            "coordination_state_cell.weight_ih", summary["missing_after_load"]
        )
        self.assertIn("actionsMLP.0.weight", summary["missing_after_load"])

        torch.testing.assert_close(
            new_model.state_dict()["cnn.convs.0.weight"],
            torch.full_like(new_model.state_dict()["cnn.convs.0.weight"], 0.25),
        )


if __name__ == "__main__":
    unittest.main()
