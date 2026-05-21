import time

from tqdm import tqdm

import torch
from loguru import logger
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse, scatter
from torch_geometric.data import Batch, Data

from hmagat.progress_logging import ProgressLogger


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


def group_indices_by_graph_map_id(graph_map_id):
    if isinstance(graph_map_id, torch.Tensor):
        graph_map_id = graph_map_id.detach().cpu().tolist()

    groups = []
    seen_map_ids = set()
    current_map_id = None
    current_group = []

    for index, map_id in enumerate(graph_map_id):
        if isinstance(map_id, torch.Tensor):
            map_id = map_id.item()

        if current_map_id is None:
            current_map_id = map_id
            current_group = [index]
            seen_map_ids.add(map_id)
            continue

        if map_id == current_map_id:
            current_group.append(index)
            continue

        groups.append(current_group)
        if map_id in seen_map_ids:
            raise ValueError(
                "graph_map_id must be contiguous for sequence grouping; "
                f"map id {map_id} appears in multiple segments."
            )
        seen_map_ids.add(map_id)
        current_map_id = map_id
        current_group = [index]

    if current_group:
        groups.append(current_group)

    return groups


def _data_num_agents(data):
    if hasattr(data, "x") and data.x is not None:
        return data.x.shape[0]
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        return data.num_nodes
    raise ValueError(
        "Cannot validate sequence assumptions: snapshot has neither x nor "
        "num_nodes to infer the number of agents."
    )


def _first_step_value(data):
    if not hasattr(data, "first_step"):
        return None
    first_step = data.first_step
    if isinstance(first_step, torch.Tensor):
        return bool(first_step.reshape(-1)[0].item())
    return bool(first_step)


def _agent_id_for_num_agents(num_agents):
    return torch.arange(num_agents, dtype=torch.long)


def validate_sequence_dataset_assumptions(
    snapshot_dataset,
    graph_map_id=None,
    require_agent_id=False,
    progress_label=None,
):
    if graph_map_id is None:
        if not hasattr(snapshot_dataset, "graph_map_id"):
            raise ValueError(
                "graph_map_id must be provided when snapshot_dataset does not "
                "expose a graph_map_id attribute."
            )
        graph_map_id = snapshot_dataset.graph_map_id

    episode_indices = group_indices_by_graph_map_id(graph_map_id)
    if progress_label is not None:
        logger.info(
            f"{progress_label}: grouped episodes={len(episode_indices)}, "
            f"snapshot_dataset_len={len(snapshot_dataset)}. Starting checks."
        )
    progress = None
    if progress_label is not None:
        progress = ProgressLogger(
            progress_label,
            len(episode_indices),
            every_n=max(1, len(episode_indices) // 20) if episode_indices else 1,
            every_seconds=30.0,
        )
    missing_agent_id_warning_emitted = False
    num_snapshots = 0

    for episode_idx, indices in enumerate(episode_indices):
        expected_num_agents = None
        expected_agent_id = None
        for timestep, snapshot_idx in enumerate(indices):
            data = snapshot_dataset[snapshot_idx]
            num_snapshots += 1

            num_agents = _data_num_agents(data)
            if expected_num_agents is None:
                expected_num_agents = num_agents
            elif num_agents != expected_num_agents:
                raise ValueError(
                    "Agent row count changed inside one sequence episode: "
                    f"episode {episode_idx}, snapshot {snapshot_idx}, expected "
                    f"{expected_num_agents}, got {num_agents}."
                )

            first_step = _first_step_value(data)
            if first_step is not None:
                expected_first_step = timestep == 0
                if first_step != expected_first_step:
                    raise ValueError(
                        "first_step flag is inconsistent with sequence grouping: "
                        f"episode {episode_idx}, snapshot {snapshot_idx}, "
                        f"expected {expected_first_step}, got {first_step}."
                    )

            if hasattr(data, "agent_id"):
                agent_id = data.agent_id.detach().cpu()
                if expected_agent_id is None:
                    expected_agent_id = agent_id
                elif not torch.equal(agent_id, expected_agent_id):
                    raise ValueError(
                        "agent_id order changed inside one sequence episode: "
                        f"episode {episode_idx}, snapshot {snapshot_idx}."
                    )
            elif require_agent_id:
                raise ValueError(
                    "Cannot prove stable agent row order: snapshot has no "
                    "agent_id attribute and require_agent_id=True."
                )
            elif not missing_agent_id_warning_emitted:
                logger.warning(
                    "Sequence dataset assumption check cannot prove stable agent "
                    "row order because snapshots have no agent_id attribute; "
                    "falling back to row-order stability assumption."
                )
                missing_agent_id_warning_emitted = True
        if progress is not None:
            progress.update(
                episode_idx + 1,
                extra=f"snapshots={num_snapshots}",
            )

    result = {
        "num_episodes": len(episode_indices),
        "num_snapshots": num_snapshots,
        "episode_lengths": [len(indices) for indices in episode_indices],
        "checked_agent_id": not missing_agent_id_warning_emitted,
    }
    if progress_label is not None:
        logger.info(
            f"{progress_label} finished: episodes={result['num_episodes']}, "
            f"snapshots={result['num_snapshots']}, "
            f"checked_agent_id={result['checked_agent_id']}"
        )
    return result


class MAPFSequenceDataset(Dataset):
    def __init__(self, snapshot_dataset, graph_map_id=None):
        self.snapshot_dataset = snapshot_dataset
        if graph_map_id is None:
            if not hasattr(snapshot_dataset, "graph_map_id"):
                raise ValueError(
                    "graph_map_id must be provided when snapshot_dataset does not "
                    "expose a graph_map_id attribute."
                )
            graph_map_id = snapshot_dataset.graph_map_id
        self.episode_indices = group_indices_by_graph_map_id(graph_map_id)

    def __len__(self) -> int:
        return len(self.episode_indices)

    def __getitem__(self, index):
        return [self.snapshot_dataset[i] for i in self.episode_indices[index]]


class MAPFSequenceBatch:
    def __init__(
        self,
        timesteps,
        active_episode_indices,
        sequence_lengths,
        *,
        active_episode_index_lists=None,
        timestep_ptr_lists=None,
        graph_counts=None,
        first_step_graph_counts=None,
        node_row_counts=None,
    ):
        self.timesteps = timesteps
        self.active_episode_indices = active_episode_indices
        self.sequence_lengths = sequence_lengths
        self.active_episode_index_lists = active_episode_index_lists
        self.timestep_ptr_lists = timestep_ptr_lists
        self.graph_counts = graph_counts
        self.first_step_graph_counts = first_step_graph_counts
        self.node_row_counts = node_row_counts

    def __len__(self):
        return len(self.timesteps)


def _graph_num_node_rows(data):
    x = getattr(data, "x", None)
    if x is not None:
        return int(x.shape[0])
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None:
        raise ValueError(
            "Cannot determine number of node rows for sequence timestep item."
        )
    return int(num_nodes)


def collate_mapf_sequences(sequences):
    if len(sequences) == 0:
        raise ValueError("Cannot collate an empty sequence batch.")

    sequence_lengths_list = [len(sequence) for sequence in sequences]
    sequence_lengths = torch.tensor(sequence_lengths_list)
    max_length = int(torch.max(sequence_lengths).item())
    if max_length == 0:
        raise ValueError("Cannot collate sequence batch with no timesteps.")

    timestep_items = [[] for _ in range(max_length)]
    timestep_active_indices = [[] for _ in range(max_length)]
    for episode_idx, sequence in enumerate(sequences):
        append_timestep_items = getattr(sequence, "_append_timestep_items", None)
        if append_timestep_items is not None:
            append_timestep_items(timestep_items, timestep_active_indices, episode_idx)
            continue
        for timestep, data in enumerate(sequence):
            timestep_items[timestep].append(data)
            timestep_active_indices[timestep].append(episode_idx)

    timesteps = []
    active_episode_indices = []
    active_episode_index_lists = []
    timestep_ptr_lists = []
    graph_counts = []
    first_step_graph_counts = []
    node_row_counts = []
    for items, indices in zip(timestep_items, timestep_active_indices):
        batch = Batch.from_data_list(items)
        timesteps.append(batch)
        active_episode_indices.append(torch.tensor(indices, dtype=torch.long))
        active_episode_index_lists.append(tuple(int(index) for index in indices))
        timestep_ptr_lists.append(tuple(int(offset) for offset in batch.ptr.tolist()))
        graph_counts.append(len(items))
        first_step_graph_counts.append(
            sum(bool(data.first_step.item()) for data in items)
        )
        node_row_counts.append(sum(_graph_num_node_rows(data) for data in items))

    return MAPFSequenceBatch(
        timesteps=timesteps,
        active_episode_indices=active_episode_indices,
        sequence_lengths=sequence_lengths,
        active_episode_index_lists=tuple(active_episode_index_lists),
        timestep_ptr_lists=tuple(timestep_ptr_lists),
        graph_counts=tuple(graph_counts),
        first_step_graph_counts=tuple(first_step_graph_counts),
        node_row_counts=tuple(node_row_counts),
    )


class SequenceBatchCollator:
    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        self.total_collate_calls = 0
        self.total_collate_time_sec = 0.0
        self.last_collate_time_sec = 0.0

    def metrics_snapshot(self):
        return {
            "collate_calls": int(self.total_collate_calls),
            "collate_time_sec": float(self.total_collate_time_sec),
            "last_collate_time_sec": float(self.last_collate_time_sec),
        }

    def __call__(self, sequences):
        start_time = time.monotonic()
        batch = collate_mapf_sequences(sequences)
        elapsed = time.monotonic() - start_time
        self.total_collate_calls += 1
        self.total_collate_time_sec += elapsed
        self.last_collate_time_sec = elapsed
        return batch


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
        extra_kwargs = extra_kwargs | {
            "first_step": first_step,
            "agent_id": _agent_id_for_num_agents(x.shape[0]),
        }

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
        extra_kwargs = extra_kwargs | {
            "first_step": first_step,
            "agent_id": _agent_id_for_num_agents(x.shape[0]),
        }

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
