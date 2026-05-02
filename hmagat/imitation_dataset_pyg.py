from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse, scatter
from torch_geometric.data import Data


def convert_dense_graph_dataset_to_sparse_pyg_dataset(dense_dataset):
    new_graph_dataset = []
    (
        dataset_node_features,
        dataset_Adj,
        dataset_target_actions,
        dataset_terminated,
        graph_map_id,
    ) = dense_dataset
    for i in tqdm(range(dataset_node_features.shape[0])):
        edge_index, edge_weight = dense_to_sparse(dataset_Adj[i])
        new_graph_dataset.append(
            Data(
                x=dataset_node_features[i],
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=dataset_target_actions[i],
                terminated=dataset_terminated[i],
            )
        )
    return new_graph_dataset, graph_map_id


def decode_dense_dataset(dense_dataset, use_edge_attr):
    if use_edge_attr:
        return dense_dataset
    return *dense_dataset, None


def get_node_features(
    node_features,
    additional_data,
    additional_data_idx,
    index,
):
    # Updating to include cost-to-go data
    node_features = node_features[index]
    cost_to_go_idx = additional_data_idx[0]
    if cost_to_go_idx is None:
        return node_features
    cost_to_go = additional_data[index][cost_to_go_idx]
    cost_to_go = torch.unsqueeze(cost_to_go, dim=1)

    return torch.cat([node_features, cost_to_go], dim=1)


def add_additional_data(additional_data, additional_data_idx, index, dtype):
    kwargs = dict()
    if additional_data_idx[1] is not None:
        kwargs["greedy_action"] = additional_data[index][additional_data_idx[1]]
    if additional_data_idx[2] is not None:
        idx, _ = additional_data_idx[2]
        prev_actions = torch.nn.functional.one_hot(
            additional_data[index][idx], num_classes=5
        )
        prev_actions = prev_actions.reshape((prev_actions.shape[0], -1))
        prev_actions = prev_actions.to(dtype)
        kwargs["prev_actions"] = prev_actions
    return kwargs


class MAPFGraphDataset(Dataset):
    def __init__(
        self,
        dense_dataset,
        use_edge_attr,
        target_vec=None,
        use_target_vec=None,
        edge_attr_opts="straight",
        additional_data=None,
        additional_data_idx=[None, None, None],
        use_edge_attr_for_messages=None,
    ) -> None:
        (
            self.dataset_node_features,
            self.dataset_Adj,
            self.dataset_target_actions,
            self.dataset_terminated,
            self.graph_map_id,
            self.dataset_agent_pos,
        ) = decode_dense_dataset(dense_dataset, use_edge_attr)
        self.use_edge_attr = use_edge_attr
        self.edge_attr_opts = edge_attr_opts
        self.target_vec = target_vec
        self.use_target_vec = use_target_vec

        self.additional_data = additional_data
        self.additional_data_idx = additional_data_idx

        self.use_edge_attr_for_messages = use_edge_attr_for_messages

        if use_edge_attr_for_messages is not None:
            assert (
                self.use_edge_attr
            ), "Need to use edge_attr to use edge_attr_for_messages."

    def __len__(self) -> int:
        return len(self.dataset_node_features)

    def get_edge_index(self, index):
        return dense_to_sparse(self.dataset_Adj[index])

    def additional_kwargs(self, index, kwargs):
        return kwargs

    def return_data_item(self, kwargs):
        return Data(**kwargs)

    def __getitem__(self, index):
        edge_index, edge_weight = self.get_edge_index(index)
        edge_attr = None
        x = get_node_features(
            node_features=self.dataset_node_features,
            additional_data=self.additional_data,
            additional_data_idx=self.additional_data_idx,
            index=index,
        )
        y = self.dataset_target_actions[index]

        target_vec = None
        if self.use_target_vec is not None:
            target_vec = self.target_vec[index].to(torch.float)

        extra_kwargs = dict()
        if self.use_edge_attr:
            agent_pos = self.dataset_agent_pos[index]
            pos_diff = agent_pos[edge_index[0]] - agent_pos[edge_index[1]]

            if self.use_edge_attr_for_messages is not None:
                if self.use_edge_attr_for_messages == "positions":
                    edge_attr = pos_diff.to(torch.float)
                elif self.use_edge_attr_for_messages == "dist":
                    edge_attr = pos_diff.to(torch.float)
                    edge_attr = torch.norm(edge_attr, keepdim=True, dim=-1)
                elif self.use_edge_attr_for_messages == "manhattan":
                    edge_attr = pos_diff.to(torch.float)
                    edge_attr = torch.sum(torch.abs(edge_attr), dim=-1, keepdim=True)
                elif self.use_edge_attr_for_messages == "positions+dist":
                    edge_attr = pos_diff.to(torch.float)
                    dist = torch.norm(edge_attr, keepdim=True, dim=-1)
                    edge_attr = torch.concatenate([edge_attr, dist], dim=-1)
                elif self.use_edge_attr_for_messages == "positions+manhattan":
                    edge_attr = pos_diff.to(torch.float)
                    manhattan = torch.sum(torch.abs(edge_attr), dim=-1, keepdim=True)
                    edge_attr = torch.concatenate([edge_attr, manhattan], dim=-1)
                else:
                    raise ValueError(
                        f"Unsupported value for use_edge_attr_for_messages: {self.use_edge_attr_for_messages}."
                    )
            else:
                edge_attr = pos_diff.to(torch.float)
                if self.edge_attr_opts == "dist":
                    dist = torch.norm(edge_attr, keepdim=True, dim=-1)
                    edge_attr = torch.concatenate([edge_attr, dist], dim=-1)
                elif self.edge_attr_opts == "only-dist":
                    edge_attr = torch.norm(edge_attr, keepdim=True, dim=-1)
                elif self.edge_attr_opts != "straight":
                    raise ValueError(
                        f"Unsupport edge_attr_opts: {self.edge_attr_opts}."
                    )
        if self.use_target_vec is not None:
            if self.use_target_vec == "target-vec+dist":
                # Calculating dist
                dist = torch.norm(target_vec, keepdim=True, dim=-1)
                target_vec = torch.concatenate([target_vec, dist], dim=-1)
            extra_kwargs["target_vec"] = target_vec

        # Adding First Step kwarg
        if index == 0:
            first_step = True
        else:
            first_step = self.graph_map_id[index] != self.graph_map_id[index - 1]
        first_step = torch.BoolTensor([first_step])
        extra_kwargs = extra_kwargs | {"first_step": first_step}

        extra_kwargs = extra_kwargs | add_additional_data(
            additional_data=self.additional_data,
            additional_data_idx=self.additional_data_idx,
            index=index,
            dtype=x.dtype,
        )
        kwargs = (
            dict(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_attr=edge_attr,
                y=y,
                terminated=self.dataset_terminated[index],
            )
            | extra_kwargs
        )

        kwargs = self.additional_kwargs(index, kwargs)

        return self.return_data_item(kwargs)


class DirectionalHypergraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_dst":
            return torch.max(value).item() + 1
        elif key == "hton_edge_index_src":
            return torch.max(value).item() + 1
        return super().__inc__(key, value, *args, **kwargs)


class MAPFHypergraphDataset(Dataset):
    def __init__(
        self,
        dense_dataset,
        hyperedge_indices,
        use_edge_attr=False,
        target_vec=None,
        use_target_vec=None,
        edge_attr_opts="straight",
        additional_data=None,
        additional_data_idx=[None, None, None],
        use_edge_attr_for_messages=None,
    ) -> None:
        (
            self.dataset_node_features,
            self.dataset_Adj,
            self.dataset_target_actions,
            self.dataset_terminated,
            self.graph_map_id,
            self.dataset_agent_pos,
        ) = decode_dense_dataset(dense_dataset, use_edge_attr)
        self.hyperedge_indices, self.hton_indices = hyperedge_indices

        self.use_edge_attr = use_edge_attr
        self.edge_attr_opts = edge_attr_opts
        self.target_vec = target_vec
        self.use_target_vec = use_target_vec
        self.additional_data = additional_data
        self.additional_data_idx = additional_data_idx

        self.use_edge_attr_for_messages = use_edge_attr_for_messages

    def __len__(self) -> int:
        return len(self.dataset_node_features)

    def __getitem__(self, index):
        extra_kwargs = dict()
        graph_edge_index, graph_edge_weight = None, None
        y = self.dataset_target_actions[index]

        x = get_node_features(
            node_features=self.dataset_node_features,
            additional_data=self.additional_data,
            additional_data_idx=self.additional_data_idx,
            index=index,
        )

        edge_index = torch.LongTensor(self.hyperedge_indices[index])
        hton_index = torch.LongTensor(self.hton_indices[index])

        if self.use_edge_attr:
            agent_pos = self.dataset_agent_pos[index]
            edge_centre_pos = agent_pos[hton_index[1]]
            edge_centre_pos = scatter(
                edge_centre_pos, hton_index[0], dim=0, reduce="mean"
            )
            pos_diff = agent_pos[edge_index[0]] - edge_centre_pos[edge_index[1]]

            if self.use_edge_attr_for_messages is not None:
                if self.use_edge_attr_for_messages == "positions":
                    edge_attr = pos_diff.to(torch.float)
                elif self.use_edge_attr_for_messages == "dist":
                    edge_attr = pos_diff.to(torch.float)
                    edge_attr = torch.norm(edge_attr, keepdim=True, dim=-1)
                elif self.use_edge_attr_for_messages == "manhattan":
                    edge_attr = pos_diff.to(torch.float)
                    edge_attr = torch.sum(torch.abs(edge_attr), dim=-1, keepdim=True)
                elif self.use_edge_attr_for_messages == "positions+dist":
                    edge_attr = pos_diff.to(torch.float)
                    dist = torch.norm(edge_attr, keepdim=True, dim=-1)
                    edge_attr = torch.concatenate([edge_attr, dist], dim=-1)
                elif self.use_edge_attr_for_messages == "positions+manhattan":
                    edge_attr = pos_diff.to(torch.float)
                    manhattan = torch.sum(torch.abs(edge_attr), dim=-1, keepdim=True)
                    edge_attr = torch.concatenate([edge_attr, manhattan], dim=-1)
                else:
                    raise ValueError(
                        f"Unsupported value for use_edge_attr_for_messages: {self.use_edge_attr_for_messages}."
                    )
                hton_edge_attr = None
                extra_kwargs = extra_kwargs | {"hton_edge_attr": hton_edge_attr}
            else:
                raise NotImplementedError("Yet to be implemented.")
            extra_kwargs = extra_kwargs | {"edge_attr": edge_attr}
        if self.use_target_vec is not None:
            target_vec = self.target_vec[index].to(torch.float)
            if self.use_target_vec == "target-vec+dist":
                # Calculating dist
                dist = torch.norm(target_vec, keepdim=True, dim=-1)
                target_vec = torch.concatenate([target_vec, dist], dim=-1)
            extra_kwargs["target_vec"] = target_vec

        if index == 0:
            first_step = True
        else:
            first_step = self.graph_map_id[index] != self.graph_map_id[index - 1]
        first_step = torch.BoolTensor([first_step])
        extra_kwargs = extra_kwargs | {"first_step": first_step}

        extra_kwargs = extra_kwargs | add_additional_data(
            additional_data=self.additional_data,
            additional_data_idx=self.additional_data_idx,
            index=index,
            dtype=x.dtype,
        )
        kwargs = (
            dict(
                x=x,
                edge_index_src=edge_index[0],
                edge_index_dst=edge_index[1],
                y=y,
                terminated=self.dataset_terminated[index],
            )
            | extra_kwargs
        )

        kwargs = kwargs | {
            "hton_edge_index_src": hton_index[0],
            "hton_edge_index_dst": hton_index[1],
        }
        return DirectionalHypergraphData(**kwargs)
