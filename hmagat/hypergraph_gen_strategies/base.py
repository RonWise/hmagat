from abc import ABC, abstractmethod
import numpy as np
import torch

from scipy.spatial.distance import squareform, pdist
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction import grid_to_graph
import pyamg
from loguru import logger

from pibt.pypibt.dist_table import DistTable


def get_unique_groups_no_hpos(groups):
    unique_groups = []
    for group in groups:
        unique = True
        for g in unique_groups:
            if g == group:
                unique = False
                break
        if unique:
            unique_groups.append(group)
    return unique_groups


def get_unique_groups(groups, group_hpos=None, hpos_aggregation="mean"):
    if group_hpos is None:
        return get_unique_groups_no_hpos(groups)
    unique_groups = []
    unique_group_hpos = []
    for group, hpos in zip(groups, group_hpos):
        unique = True
        for i, g in enumerate(unique_groups):
            if g == group:
                unique = False
                unique_group_hpos[i].append(hpos)
                break
        if unique:
            unique_groups.append(group)
            unique_group_hpos.append([hpos])
    if hpos_aggregation == "mean":
        unique_group_hpos = [
            torch.mean(torch.stack(hpos), dim=0) for hpos in unique_group_hpos
        ]
        unique_group_hpos = torch.stack(unique_group_hpos)
    else:
        raise ValueError(f"Unknown hpos_aggregation: {hpos_aggregation}.")
    return unique_groups, unique_group_hpos


class HyperedgeGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset_state(self, env=None):
        pass

    @abstractmethod
    def __call__(self, env):
        pass


class ShortestDistanceAllCliquesHyperedgeGenerator(HyperedgeGenerator):
    def __init__(
        self,
        comm_radius,
        distance_metric="euclidean",
        comm_self=True,
        hypergraph_max_neighbours=None,
        neighbour_cutoff_method="closest",
        max_dist_threshold=3,
        max_dist_frac=0.8,
        max_clique_size=4,
        add_hypergraph_self_loop=False,
    ):
        self.comm_radius = comm_radius
        self.distance_metric = distance_metric
        self.comm_self = comm_self
        self.hypergraph_max_neighbours = hypergraph_max_neighbours
        self.neighbour_cutoff_method = neighbour_cutoff_method
        self.max_dist_threshold = max_dist_threshold
        self.max_dist_frac = max_dist_frac
        self.max_clique_size = max_clique_size
        self.add_hypergraph_self_loop = add_hypergraph_self_loop

    def reset_state(self, env):
        self.move_results = np.array(env.grid_config.MOVES)
        self.grid = env.grid.get_obstacles(ignore_borders=True) == 0

    def get_cliques(self, allowed_pairs):
        all_cliques = []

        # Start with just single nodes
        cliques = []
        for i in range(len(allowed_pairs)):
            a = np.zeros(len(allowed_pairs), dtype=bool)
            a[i] = True
            cliques.append(a)

        while len(cliques) > 0:
            new_cliques = []
            np_nc = None
            k = np.sum(cliques[0])
            if k >= self.max_clique_size:
                all_cliques.extend(cliques)
                break
            for i in range(len(cliques)):
                found_match = False
                for j in range(len(cliques)):
                    if i == j:
                        continue
                    intersec = cliques[i] * cliques[j]
                    if np.sum(intersec) == k - 1:
                        union = cliques[i] + cliques[j]
                        if len(new_cliques) > 0:
                            if np.any(np.all(np_nc == union, axis=-1)):
                                # Already exists
                                found_match = True
                                continue
                        xor = union * (~intersec)
                        edge = np.nonzero(xor)[0]
                        if allowed_pairs[edge[0], edge[1]]:
                            # Adding new clique
                            found_match = True
                            new_cliques.append(union)
                            np_nc = np.stack(new_cliques, axis=0)
                if not found_match:
                    all_cliques.append(cliques[i])
            cliques = new_cliques

        return all_cliques

    def __call__(self, env):
        agent_pos = env.grid.get_agents_xy(ignore_borders=True)

        pos = [tuple(s) for s in agent_pos]
        dtables = [DistTable(self.grid, p) for p in pos]

        agent_pos = np.array(agent_pos)
        pos_diff = agent_pos[None] - agent_pos[:, None]
        if self.distance_metric == "euclidean":
            pos_diff = np.sum(pos_diff**2, axis=-1) ** 0.5
        elif self.distance_metric == "manhattan":
            pos_diff = np.sum(np.abs(pos_diff), axis=-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}.")

        mask = pos_diff <= self.comm_radius
        if not self.comm_self:
            np.fill_diagonal(mask, False)
        if self.hypergraph_max_neighbours is not None:
            pos_diff = mask * pos_diff
            if self.neighbour_cutoff_method == "closest":
                Adj_temp = pos_diff + np.where(~mask, np.inf, 0.0)
                idx = np.argsort(Adj_temp, axis=-1)
            elif self.neighbour_cutoff_method == "random":
                vals = mask * np.random.rand(*pos_diff.shape)
                idx = np.argsort(-vals, axis=-1)
            else:
                raise ValueError(
                    f"Unsupported neighbour_cutoff_method: {self.neighbour_cutoff_method}."
                )
            idx = idx[:, self.hypergraph_max_neighbours :]
            np.put_along_axis(mask, idx, values=False, axis=-1)
        pos_diff = mask

        edge_index_src = []
        edge_index_dst = []
        hton_index_src = []
        hton_index_dst = []

        hyperedge_offset = 0

        for agent_idx in range(len(pos)):
            in_edges = np.nonzero(pos_diff[agent_idx])[0]
            if len(in_edges) == 0:
                continue

            u_to_v = np.zeros((len(in_edges), len(in_edges)))
            for i, e1 in enumerate(in_edges):
                for j, e2 in enumerate(in_edges):
                    u_to_v[i, j] = dtables[e1].get(pos[e2])

            a_to_u = np.zeros(len(in_edges))
            for i, e in enumerate(in_edges):
                a_to_u[i] = dtables[agent_idx].get(pos[e])

            a_to_u, a_to_v = a_to_u[None], a_to_u[:, None]
            a_to_u_to_v = (
                a_to_u + u_to_v
            )  # a_to_u_to_v[v, u] is dist from a to u + u to v

            dist_diff = a_to_u_to_v - a_to_v
            dist_diff = np.minimum(dist_diff, dist_diff.T)

            base_dist = np.minimum(a_to_u, a_to_v)
            base_dist = np.where(
                base_dist > self.max_dist_threshold,
                base_dist * self.max_dist_frac,
                base_dist,
            )

            allowed_pairs = dist_diff <= base_dist

            all_cliques = self.get_cliques(allowed_pairs)
            all_cliques = np.array(all_cliques)

            cur_edge_index_src = np.tile(in_edges, len(all_cliques))
            cur_edge_index_src = cur_edge_index_src[all_cliques.flatten()]

            cur_edge_index_dst = hyperedge_offset + np.arange(len(all_cliques)).repeat(
                len(in_edges)
            )
            cur_edge_index_dst = cur_edge_index_dst[all_cliques.flatten()]

            cur_hton_index_src = hyperedge_offset + np.arange(len(all_cliques))
            cur_hton_index_dst = np.ones_like(cur_hton_index_src) * agent_idx

            hyperedge_offset += len(all_cliques)
            edge_index_src.append(cur_edge_index_src)
            edge_index_dst.append(cur_edge_index_dst)
            hton_index_src.append(cur_hton_index_src)
            hton_index_dst.append(cur_hton_index_dst)
        edge_index_src = np.concatenate(edge_index_src, axis=0)
        edge_index_dst = np.concatenate(edge_index_dst, axis=0)
        hton_index_src = np.concatenate(hton_index_src, axis=0)
        hton_index_dst = np.concatenate(hton_index_dst, axis=0)

        if self.add_hypergraph_self_loop:
            loop = np.arange(agent_pos.shape[1])
            num_existing_edges = hyperedge_offset
            edge_index_src = np.concatenate([edge_index_src, loop], axis=0)
            edge_index_dst = np.concatenate(
                [edge_index_dst, num_existing_edges + loop], axis=0
            )
            hton_index_src = np.concatenate(
                [hton_index_src, num_existing_edges + loop], axis=0
            )
            hton_index_dst = np.concatenate([hton_index_dst, loop], axis=0)

        edge_index = [edge_index_src.tolist(), edge_index_dst.tolist()]
        hton_index = [hton_index_src.tolist(), hton_index_dst.tolist()]

        return edge_index, hton_index


class ShortestDistanceSampleCliquesHyperedgeGenerator(
    ShortestDistanceAllCliquesHyperedgeGenerator
):
    def __init__(
        self,
        comm_radius,
        distance_metric="euclidean",
        comm_self=True,
        hypergraph_max_neighbours=None,
        neighbour_cutoff_method="closest",
        max_dist_threshold=3,
        max_dist_frac=0.8,
        max_clique_size=4,
        add_hypergraph_self_loop=False,
        seed=42,
        recommended_hyperedges=5,
    ):
        super().__init__(
            comm_radius,
            distance_metric,
            comm_self,
            hypergraph_max_neighbours,
            neighbour_cutoff_method,
            max_dist_threshold,
            max_dist_frac,
            max_clique_size,
            add_hypergraph_self_loop,
        )
        self.seed = seed
        self.recommended_hyperedges = recommended_hyperedges

    def reset_state(self, env):
        super().reset_state(env)
        self.rng = np.random.default_rng(self.seed)

    def get_cliques(self, allowed_pairs):
        all_cliques = []

        unvisited = np.ones(len(allowed_pairs), dtype=bool)
        while np.sum(unvisited) > 0:
            choice1 = self.rng.choice(np.nonzero(unvisited)[0])

            next_clique = np.zeros(len(allowed_pairs), dtype=bool)
            next_clique[choice1] = True

            choices = allowed_pairs[choice1]
            choices = choices * (~next_clique)

            while np.sum(choices) > 0:
                choice2 = self.rng.choice(np.nonzero(choices)[0])
                next_clique[choice2] = True

                choices = choices * allowed_pairs[choice2]
                choices = choices * (~next_clique)

            all_cliques.append(next_clique)
            unvisited = unvisited * (~next_clique)

        np_nc = np.stack(all_cliques, axis=0)

        for _ in range(self.recommended_hyperedges - len(all_cliques)):
            choice1 = self.rng.integers(len(allowed_pairs))

            next_clique = np.zeros(len(allowed_pairs), dtype=bool)
            next_clique[choice1] = True

            choices = allowed_pairs[choice1]
            choices = choices * (~next_clique)

            while np.sum(choices) > 0:
                choice2 = self.rng.choice(np.nonzero(choices)[0])
                next_clique[choice2] = True

                choices = choices * allowed_pairs[choice2]
                choices = choices * (~next_clique)

            if np.any(np.all(np_nc == next_clique, axis=-1)):
                # Already exists
                pass
            else:
                all_cliques.append(next_clique)
                np_nc = np.stack(all_cliques, axis=0)
        return all_cliques


class kMeansHyperedgeGenerator(HyperedgeGenerator):
    def __init__(
        self,
        comm_radius,
        distance_metric="euclidean",
        comm_self=True,
        hypergraph_max_neighbours=None,
        neighbour_cutoff_method="closest",
        add_hypergraph_self_loop=False,
        initial_colour_percentage=0.1,
        final_colour_percentage=0.1,
        only_wait_for_atleast_one_colour=False,
        diameter_based_num_colours=False,
        num_updates=100,
        seed=42,
    ):
        self.comm_radius = comm_radius
        self.distance_metric = distance_metric
        self.comm_self = comm_self
        self.hypergraph_max_neighbours = hypergraph_max_neighbours
        self.neighbour_cutoff_method = neighbour_cutoff_method
        self.add_hypergraph_self_loop = add_hypergraph_self_loop
        self.initial_colour_percentage = initial_colour_percentage
        self.final_colour_percentage = final_colour_percentage
        self.only_wait_for_atleast_one_colour = only_wait_for_atleast_one_colour
        self.diameter_based_num_colours = diameter_based_num_colours
        self.num_updates = num_updates
        self.seed = seed

    def reset_state(self, env):
        self.grid = env.grid.get_obstacles(ignore_borders=True)
        self.rng = np.random.default_rng(self.seed)
        self.colourings = self.colour_grid()

    def update_colours(self, colours, obs):
        new_colours = colours.copy()
        new_colours[:-1] += colours[1:]
        new_colours[1:] += colours[:-1]
        new_colours[:, :-1] += colours[:, 1:]
        new_colours[:, 1:] += colours[:, :-1]

        sum_colours = np.sum(new_colours, axis=-1, keepdims=True)
        sum_colours = np.where(sum_colours, sum_colours, 1)
        new_colours /= sum_colours

        new_colours[obs == 1] = colours[obs == 1]

        return new_colours

    def get_diameter(self):
        initial_vertex = tuple(np.argwhere(self.grid == 0)[0])

        def get_furthest_node_and_distance(vertex):
            dists = -np.ones(self.grid.shape, dtype=int)
            dists[vertex[0], vertex[1]] = 0

            Q = [vertex]

            while Q:
                v = Q.pop(0)
                neighbours = [
                    (v[0] + 1, v[1]),
                    (v[0] - 1, v[1]),
                    (v[0], v[1] + 1),
                    (v[0], v[1] - 1),
                ]
                for n in neighbours:
                    if (
                        n[0] < 0
                        or n[0] >= self.grid.shape[0]
                        or n[1] < 0
                        or n[1] >= self.grid.shape[1]
                    ):
                        continue
                    if self.grid[n] == 1 or dists[n] != -1:
                        continue
                    dists[n] = dists[v] + 1
                    Q.append(n)
            return v, dists[v]

        furthest_node, _ = get_furthest_node_and_distance(initial_vertex)
        _, diameter = get_furthest_node_and_distance(furthest_node)
        return diameter

    def colour_grid(self):
        obs = self.grid
        starting_points = np.argwhere(obs == 0)

        if self.diameter_based_num_colours:
            diameter = self.get_diameter()
            num_colours = int(diameter * self.initial_colour_percentage)
            num_final_colours = int(diameter * self.final_colour_percentage)

            num_colours = max(num_colours, 2)
            num_final_colours = max(num_final_colours, 2)
        else:
            num_colours = int(len(starting_points) * self.initial_colour_percentage)
            num_final_colours = int(len(starting_points) * self.final_colour_percentage)
        colours = np.zeros((*obs.shape, num_colours))

        starting_points = self.rng.permutation(starting_points)
        for i in range(num_colours):
            x, y = starting_points[i]
            colours[x, y, i] = 1

        for _ in range(self.num_updates):
            colours = self.update_colours(colours, obs)

        clusterer = KMeans(n_clusters=2 * num_final_colours, max_iter=100).fit(
            colours[obs == 0]
        )

        one_hot_colouring = np.eye(2 * num_final_colours)
        one_hot_groupings = one_hot_colouring[clusterer.labels_]

        num_groupings = np.sum(one_hot_groupings, axis=0)
        grouping_order = np.argsort(num_groupings)

        colourings = np.zeros((*obs.shape, num_final_colours))
        colourings[obs == 0] = one_hot_groupings[:, grouping_order[-num_final_colours:]]

        mask = np.sum(colourings, axis=-1) + obs
        for _ in range(int(np.sum(1 - mask))):
            if self.only_wait_for_atleast_one_colour and np.all(
                np.sum(colourings, axis=-1) + obs > 0
            ):
                break
            colourings = self.update_colours(colourings, mask)
        colourings = colourings > 0

        return colourings

    def get_groups(self, relevant_colourings):
        groups = []
        for i in range(relevant_colourings.shape[1]):
            group = set(np.nonzero(relevant_colourings[:, i])[0])
            groups.append(group)

        return get_unique_groups_no_hpos(groups)

    def get_agent_colourings(self, agent_id, agent_comm_mask, agent_pos):
        comm_agent_pos = agent_pos[agent_comm_mask[agent_id]]
        return self.colourings[comm_agent_pos[:, 0], comm_agent_pos[:, 1]]

    def __call__(self, env):
        agent_pos = env.grid.get_agents_xy(ignore_borders=True)
        agent_pos = np.array(agent_pos)

        pos_diff = agent_pos[None] - agent_pos[:, None]
        if self.distance_metric == "euclidean":
            pos_diff = np.sum(pos_diff**2, axis=-1) ** 0.5
        elif self.distance_metric == "manhattan":
            pos_diff = np.sum(np.abs(pos_diff), axis=-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}.")

        mask = pos_diff <= self.comm_radius
        if not self.comm_self:
            np.fill_diagonal(mask, False)
        if self.hypergraph_max_neighbours is not None:
            pos_diff = mask * pos_diff
            if self.neighbour_cutoff_method == "closest":
                Adj_temp = pos_diff + np.where(~mask, np.inf, 0.0)
                idx = np.argsort(Adj_temp, axis=-1)
            elif self.neighbour_cutoff_method == "random":
                vals = mask * np.random.rand(*pos_diff.shape)
                idx = np.argsort(-vals, axis=-1)
            else:
                raise ValueError(
                    f"Unsupported neighbour_cutoff_method: {self.neighbour_cutoff_method}."
                )
            idx = idx[:, self.hypergraph_max_neighbours :]
            np.put_along_axis(mask, idx, values=False, axis=-1)
        pos_diff = mask

        if self.comm_self:
            # We will add self comm to all hyperedges later
            np.fill_diagonal(pos_diff, False)
        else:
            raise NotImplementedError

        edge_index_src = []
        edge_index_dst = []
        hton_index_src = []
        hton_index_dst = []

        hyperedge_offset = 0

        for i in range(len(agent_pos)):
            mapping = np.arange(len(agent_pos))[pos_diff[i]]

            relevant_colourings = self.get_agent_colourings(
                agent_id=i, agent_comm_mask=pos_diff, agent_pos=agent_pos
            )

            hyperedges = np.sum(relevant_colourings, axis=0) > 0
            relevant_colourings = relevant_colourings[:, hyperedges]
            groups = self.get_groups(relevant_colourings)

            for gid, group in enumerate(groups):
                for n in group:
                    edge_index_src.append(mapping[n])
                    edge_index_dst.append(hyperedge_offset + gid)

                if self.comm_self:
                    edge_index_src.append(i)
                    edge_index_dst.append(hyperedge_offset + gid)

                hton_index_src.append(hyperedge_offset + gid)
                hton_index_dst.append(i)

            hyperedge_offset += len(groups)
            if self.add_hypergraph_self_loop:
                edge_index_src.append(i)
                edge_index_dst.append(hyperedge_offset)
                hton_index_src.append(hyperedge_offset)
                hton_index_dst.append(i)
                hyperedge_offset += 1

        edge_index = [edge_index_src, edge_index_dst]
        hton_index = [hton_index_src, hton_index_dst]

        return edge_index, hton_index


class LloydsHyperedgeGenerator(kMeansHyperedgeGenerator):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def colour_grid(self):
        obs = self.grid
        starting_points = np.argwhere(obs == 0)

        num_colours = int(len(starting_points) * self.initial_colour_percentage)
        num_final_colours = int(len(starting_points) * self.final_colour_percentage)
        colours = np.zeros((*obs.shape, num_colours))

        mat = grid_to_graph(
            *obs.shape,
            mask=(1 - obs).astype(bool),
            return_as=sparse.csr_matrix,
            dtype=np.float64,
        )
        try:
            cols, centres = pyamg.graph.balanced_lloyd_cluster(
                mat, centers=num_colours, maxiter=self.num_updates
            )
        except Exception as exc:
            # Fallback to unbalanced Lloyd's clustering
            # (likely due to unconnected components)
            logger.warning(
                "Falling back to unbalanced Lloyd's clustering after balanced "
                f"Lloyd clustering failed: {exc}."
            )
            cols, centres = pyamg.graph.lloyd_cluster(
                mat, centers=num_colours, maxiter=self.num_updates
            )

        one_hot_colouring = np.eye(num_colours)
        one_hot_groupings = one_hot_colouring[cols]

        idx = np.argwhere(obs == 0).T
        colours[idx[0], idx[1]] = one_hot_groupings

        num_elems_per_element = np.sum(colours, axis=(0, 1))
        colour_idx = np.argsort(num_elems_per_element)[-num_final_colours:]
        colours = colours[:, :, colour_idx]

        mask = np.sum(colours, axis=-1) + obs
        for _ in range(int(np.sum(1 - mask))):
            if self.only_wait_for_atleast_one_colour and np.all(
                np.sum(colours, axis=-1) + obs > 0
            ):
                break
            colours = self.update_colours(colours, mask)
        colours = colours > 0

        return colours


class InfrequentHyperedgeGenerator(HyperedgeGenerator):
    def __init__(self, hyperedge_generator: HyperedgeGenerator, time_period=5):
        self.hyperedge_generator = hyperedge_generator
        self.time_period = time_period
        self.last_update = time_period
        self.previous_hyperedge = None

    def reset_state(self, env):
        self.hyperedge_generator.reset_state(env)
        self.last_update = self.time_period
        self.previous_hyperedge = None

    def __call__(self, env):
        if self.last_update >= self.time_period:
            self.previous_hyperedge = self.hyperedge_generator(env)
            self.last_update = 0

        self.last_update += 1
        return self.previous_hyperedge
