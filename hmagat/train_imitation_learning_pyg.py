import argparse
import pickle
import pathlib
import numpy as np
import wandb
import time
import random

import multiprocessing as mp
from itertools import compress
from collections import OrderedDict

import torch
import torch.optim as optim

from torch_geometric.loader import DataLoader

from hmagat.training_args import add_training_args
from hmagat.convert_to_imitation_dataset import (
    add_imitation_dataset_args,
    generate_graph_dataset,
    get_imitation_dataset_file_name,
)
from hmagat.generate_hypergraphs import (
    add_hypergraph_generation_args,
    get_hypergraph_indices_generator,
    get_hypergraph_file_name,
)
from hmagat.generate_pos import get_pos_file_name
from hmagat.run_expert import (
    get_expert_algorithm_and_config,
    run_expert_algorithm,
    add_expert_dataset_args,
)
from hmagat.imitation_dataset_pyg import (
    MAPFGraphDataset,
    MAPFHypergraphDataset,
)

from hmagat.modules.model.run_model import run_model_on_grid
from grid_config_generator import (
    grid_config_generator_factory,
    generate_grid_config_from_env,
)

from hmagat.loss import get_loss_function

from hmagat.generate_additional_data import (
    get_additional_data_file_name,
    add_additional_data_args,
    any_additional_data,
    generate_additional_data,
)

from hmagat.generate_expert_makespans import check_or_create_expert_makespans

from hmagat.lr_scheduler import get_lr_scheduler
from hmagat.dataset_loading import load_dataset


class HyperedgeIndicesGenerator:
    def __init__(
        self,
        hypergraph_comm_radius,
        max_group_size,
        hyperedge_generation_method,
        comm_self,
        hypergraph_max_neighbours,
        max_dist_threshold,
        max_dist_frac,
        max_clique_size,
        initial_colour_percentage,
        final_colour_percentage,
        only_wait_for_atleast_one_colour,
        add_hypergraph_self_loop,
        hypergraph_num_updates,
        hypergraph_time_period,
    ):
        self.hyperedge_generator = get_hypergraph_indices_generator(
            hypergraph_comm_radius=hypergraph_comm_radius,
            max_group_size=max_group_size,
            hyperedge_generation_method=hyperedge_generation_method,
            comm_self=comm_self,
            hypergraph_max_neighbours=hypergraph_max_neighbours,
            max_dist_threshold=max_dist_threshold,
            max_dist_frac=max_dist_frac,
            max_clique_size=max_clique_size,
            initial_colour_percentage=initial_colour_percentage,
            final_colour_percentage=final_colour_percentage,
            only_wait_for_atleast_one_colour=only_wait_for_atleast_one_colour,
            add_hypergraph_self_loop=add_hypergraph_self_loop,
            hypergraph_num_updates=hypergraph_num_updates,
            hypergraph_time_period=hypergraph_time_period,
        )
        self.initialized = False

    def __call__(self, env, observations, actions):
        if not self.initialized:
            self.hyperedge_generator.reset_state(env)
            self.initialized = True
        return self.hyperedge_generator(env)


def aux_func(env, observations, actions, **kwargs):
    if actions is None:
        aux_func.original_pos = np.array([obs["global_xy"] for obs in observations])
        aux_func.makespan = 0
        aux_func.costs = np.ones(env.get_num_agents())
    else:
        new_pos = np.array([obs["global_xy"] for obs in observations])
        at_goals = np.array(env.was_on_goal)
        aux_func.makespan += 1
        aux_func.original_pos = new_pos
        aux_func.costs[~at_goals] = aux_func.makespan + 1


def aux_func_train(env, observations, actions, oe_period=None, **kwargs):
    if oe_period is not None:
        aux_func_train.oe_period = oe_period
        return
    if actions is None:
        aux_func_train.grid_configs = []
        aux_func_train.original_pos = np.array(
            [obs["global_xy"] for obs in observations]
        )
        aux_func_train.makespan = 0
        aux_func_train.costs = np.ones(env.get_num_agents())
    else:
        new_pos = np.array([obs["global_xy"] for obs in observations])
        at_goals = np.array(env.was_on_goal)
        aux_func_train.makespan += 1
        aux_func_train.original_pos = new_pos
        aux_func_train.costs[~at_goals] = aux_func_train.makespan + 1
        if aux_func_train.makespan % aux_func_train.oe_period == 0:
            aux_func_train.grid_configs.append(generate_grid_config_from_env(env))


def main():
    parser = argparse.ArgumentParser(description="Train imitation learning model.")
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_additional_data_args(parser)
    parser = add_training_args(parser)

    parser.add_argument("--wandb_project", type=str, default="hyper-mapf-train")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    print(args)

    assert args.save_termination_state
    assert args.train_on_terminated_agents

    if args.device == -1:
        device = torch.device("cuda")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    stack_with_np = not args.use_lists

    rng = np.random.default_rng(args.dataset_seed)
    seeds = rng.integers(10**10, size=args.num_samples)

    _grid_config_generator = grid_config_generator_factory(args)

    grid_config = _grid_config_generator(seeds[0], map_id=0)

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    random.seed(args.model_seed)

    from hmagat.modules.agents import get_model, load_partial_state_dict

    model, hypergraph_model, dataset_kwargs = get_model(args, device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr_start, weight_decay=args.weight_decay
    )

    dense_dataset = None
    hyper_edge_indices = None
    additional_data = None

    # TODO: Update the dataset loading for all functions
    print("Loading Dataset.............")
    dense_dataset = load_dataset(
        [get_imitation_dataset_file_name],
        "processed_dataset",
        args,
    )
    if args.load_positions_separately:
        print("Loading Agent Positions.....")
        agent_pos = load_dataset(
            [get_pos_file_name],
            "positions",
            args,
        )
        dense_dataset = (*dense_dataset, agent_pos)
    if hypergraph_model:
        print("Loading Hypergraphs.........")
        hyper_edge_indices = load_dataset(
            [get_hypergraph_file_name], "hypergraphs", args
        )

    load_additional_data, additional_data_idx = any_additional_data(args)
    if load_additional_data:
        print("Loading Additional Data.....")
        additional_data = load_dataset(
            [get_additional_data_file_name], "additional_data", args
        )

    expert_makespans = None
    if args.oe_improve_quality:
        expert_makespans = check_or_create_expert_makespans(args)

    loss_function = get_loss_function(args)

    use_wandb = args.wandb_entity is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
            entity=args.wandb_entity,
        )

    unique_map_ids = np.sort(np.unique(dense_dataset[4]))
    num_samples = len(unique_map_ids)

    # Data split
    train_id_max = int(
        num_samples * (1 - args.validation_fraction - args.test_fraction)
    )
    validation_id_max = train_id_max + int(num_samples * args.validation_fraction)

    train_id_max = min(train_id_max, num_samples)
    train_id_max = unique_map_ids[train_id_max]

    validation_id_max = min(validation_id_max, num_samples)
    validation_id_max = unique_map_ids[validation_id_max]

    def _divide_dataset(start, end):
        map_ids = dense_dataset[4]
        if not isinstance(map_ids, torch.Tensor):
            map_ids = torch.from_numpy(np.array(map_ids))
        mask = torch.logical_and(map_ids >= start, map_ids < end)
        hindices, add_data = None, None
        if hyper_edge_indices is not None:
            hindices, hton_indices = hyper_edge_indices
            hindices = list(compress(hindices, mask))
            hton_indices = list(compress(hton_indices, mask))
            hindices = (hindices, hton_indices)
        if additional_data is not None:
            add_data = list(compress(additional_data, mask))
        if isinstance(dense_dataset[0], torch.Tensor):
            ds = tuple(gd[mask] for gd in dense_dataset)
        else:
            ds = tuple(list(compress(gd, mask)) for gd in dense_dataset)
        return (
            ds,
            hindices,
            add_data,
        )

    (train_dataset, train_hindices, train_additional_data) = _divide_dataset(
        0, train_id_max
    )
    (validation_dataset, validation_hindices, validation_additional_data) = (
        _divide_dataset(train_id_max, validation_id_max)
    )
    # test_dataset = _divide_dataset(validation_id_max, torch.inf)

    common_dataset_kwargs = dict(
        edge_attr_opts=args.edge_attr_opts,
        additional_data_idx=additional_data_idx,
        **dataset_kwargs,
    )

    train_kwargs = dict(additional_data=train_additional_data)
    validation_kwargs = dict(additional_data=validation_additional_data)

    if hypergraph_model:
        train_dataset = MAPFHypergraphDataset(
            train_dataset,
            train_hindices,
            **train_kwargs,
            **common_dataset_kwargs,
        )
        validation_dataset = MAPFHypergraphDataset(
            validation_dataset,
            validation_hindices,
            **validation_kwargs,
            **common_dataset_kwargs,
        )
    else:
        train_dataset = MAPFGraphDataset(
            train_dataset,
            **train_kwargs,
            **common_dataset_kwargs,
        )
        validation_dataset = MAPFGraphDataset(
            validation_dataset,
            **validation_kwargs,
            **common_dataset_kwargs,
        )
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size)
    validation_dl = DataLoader(validation_dataset, batch_size=args.batch_size)

    if expert_makespans is not None:
        expert_makespans = expert_makespans[:train_id_max]

    best_validation_success_rate = 0.0
    best_validation_accuracy = 0.0
    best_val_file_name = "best_low_val.pt"
    best_val_acc_file_name = "best_acc_val.pt"
    checkpoint_path = pathlib.Path(f"{args.checkpoints_dir}", best_val_file_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cur_validation_id_max = min(train_id_max + args.initial_val_size, validation_id_max)

    oe_graph_dataset = None
    oe_grid_configs = []
    oe_hypergraph_indices = ([], [])
    oe_additional_data = None
    oe_improve_quality = args.oe_improve_quality_threshold == 0.0

    def multiprocess_run_expert(
        queue,
        done_event,
        expert,
        grid_config,
        save_termination_state,
        hypergraph_generator=None,
    ):
        if hypergraph_generator is not None:
            hypergraph_generator.initialized = False

        expert_results = run_expert_algorithm(
            expert,
            grid_config=grid_config,
            save_termination_state=save_termination_state,
            additional_data_func=hypergraph_generator,
        )
        queue.put((*expert_results, grid_config))
        if done_event is not None:
            done_event.wait()

    hypergraph_generator = None
    if hypergraph_model:
        hypergraph_generator = HyperedgeIndicesGenerator(
            hypergraph_comm_radius=args.hypergraph_comm_radius,
            max_group_size=args.hypergraph_max_group_size,
            hyperedge_generation_method=args.hyperedge_generation_method,
            comm_self=args.comm_self,
            hypergraph_max_neighbours=args.hypergraph_max_neighbours,
            max_dist_threshold=args.hypergraph_max_dist_threshold,
            max_dist_frac=args.hypergraph_max_dist_frac,
            max_clique_size=args.hypergraph_max_clique_size,
            initial_colour_percentage=args.hypergraph_initial_colperc,
            final_colour_percentage=args.hypergraph_final_colperc,
            only_wait_for_atleast_one_colour=args.hypergraph_wait_one,
            add_hypergraph_self_loop=args.add_hypergraph_self_loop,
            hypergraph_num_updates=args.hypergraph_num_updates,
            hypergraph_time_period=args.hypergraph_time_period,
        )

    lr_scheduler = get_lr_scheduler(
        args, optimizer=optimizer, train_dataloader=train_dl
    )

    if args.pretrain_weights_path is not None:
        print("Loading Weights.............")
        pretrain_path = pathlib.Path(args.pretrain_weights_path)
        state_dict = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(state_dict)

    if args.load_partial_parameters_path is not None:
        print("Partially Loading Weights.............")
        partial_path = pathlib.Path(args.load_partial_parameters_path)
        state_dict = torch.load(partial_path, map_location=device)
        load_partial_state_dict(model, state_dict, print_prefix="[partial-load]")

    queue = mp.Queue()
    done_event = mp.Event()

    print("Starting Training....")
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        accuracies = None
        num_samples = 0
        n_batches = 0
        n_graphs = 0
        n_maps = 0

        model = model.train()
        for data in train_dl:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data)
            loss = loss_function(out, data, model)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            new_acc = loss_function.get_accuracies(out, data, model)
            if accuracies is None:
                accuracies = new_acc
            else:
                for key in accuracies:
                    accuracies[key] += new_acc[key]
            num_samples += data.x.shape[0]
            n_batches += 1
            n_graphs += len(data.ptr) - 1
            n_maps += torch.sum(data.first_step).cpu().item()
            lr_scheduler.step_on_batch()

        if oe_graph_dataset is not None:
            oe_dataset_kwargs = dict(additional_data=oe_additional_data)
            if hypergraph_model:
                oe_dl = DataLoader(
                    MAPFHypergraphDataset(
                        oe_graph_dataset,
                        oe_hypergraph_indices,
                        **oe_dataset_kwargs,
                        **common_dataset_kwargs,
                    ),
                    batch_size=args.batch_size,
                )
            else:
                oe_dl = DataLoader(
                    MAPFGraphDataset(
                        oe_graph_dataset,
                        **oe_dataset_kwargs,
                        **common_dataset_kwargs,
                    ),
                    batch_size=args.batch_size,
                )

            for data in oe_dl:
                data = data.to(device)
                optimizer.zero_grad()

                out = model(data.x, data)
                loss = loss_function(out, data, model)

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                new_acc = loss_function.get_accuracies(out, data, model)
                for key in accuracies:
                    accuracies[key] += new_acc[key]
                num_samples += data.x.shape[0]
                n_batches += 1
                n_graphs += len(data.ptr) - 1
                n_maps += torch.sum(data.first_step).cpu().item()
                lr_scheduler.step_on_batch()
        lr_scheduler.step_on_epoch()

        for key in accuracies:
            if "first_step_first_agent" in key:
                accuracies[key] = accuracies[key] / n_maps
            elif "first_agent" in key:
                accuracies[key] = accuracies[key] / n_graphs
            else:
                accuracies[key] = accuracies[key] / num_samples

        print(
            f"Epoch {epoch}, Mean Loss: {total_loss / n_batches}, Mean Accuracy: {accuracies['train_accuracy']}"
        )

        results = {"train_loss": total_loss / n_batches} | accuracies
        if (not args.skip_validation) and (
            (epoch + 1) % args.validation_every_epochs == 0
        ):
            model = model.eval()

            num_completed = 0
            all_makespan = []
            all_partial_success_rate = []
            all_sum_of_costs = []

            print("-------------------")
            print("Starting Validation")

            if not args.skip_validation_accuracy:
                val_accuracies = None
                val_samples = 0
                n_graphs = 0
                n_maps = 0

                with torch.no_grad():
                    for data in validation_dl:
                        data = data.to(device)
                        out = model(data.x, data)
                        new_acc = loss_function.get_accuracies(
                            out, data, model, "validation"
                        )

                        if val_accuracies is None:
                            val_accuracies = new_acc
                        else:
                            for key in val_accuracies:
                                val_accuracies[key] += new_acc[key]
                        val_samples += data.x.shape[0]
                        n_graphs += len(data.ptr) - 1
                        n_maps += torch.sum(data.first_step).cpu().item()
                for key in val_accuracies:
                    if "first_step_first_agent" in key:
                        val_accuracies[key] = val_accuracies[key] / n_maps
                    elif "first_agent" in key:
                        val_accuracies[key] = val_accuracies[key] / n_graphs
                    else:
                        val_accuracies[key] = val_accuracies[key] / val_samples

                val_accuracy = val_accuracies["validation_accuracy"]
                results = results | val_accuracies
                if val_accuracy > best_validation_accuracy:
                    best_validation_accuracy = val_accuracy
                    checkpoint_path = pathlib.Path(
                        args.checkpoints_dir, best_val_acc_file_name
                    )
                    torch.save(model.state_dict(), checkpoint_path)

            for graph_id in range(train_id_max, cur_validation_id_max):
                success, env, observations = run_model_on_grid(
                    model=model,
                    device=device,
                    grid_config=_grid_config_generator(
                        seeds[graph_id], map_id=graph_id
                    ),
                    args=args,
                    dataset_kwargs=dataset_kwargs,
                    hypergraph_model=hypergraph_model,
                    aux_func=aux_func,
                )

                makespan = aux_func.makespan
                costs = aux_func.costs

                partial_success_rate = np.mean(env.was_on_goal)
                sum_of_costs = np.sum(costs)

                all_makespan.append(makespan)
                all_partial_success_rate.append(partial_success_rate)
                all_sum_of_costs.append(sum_of_costs)

                if success:
                    num_completed += 1
                print(
                    f"Validation Graph {graph_id - train_id_max}/{validation_id_max - train_id_max}, "
                    f"Current Success Rate: {num_completed / (graph_id - train_id_max + 1)}"
                )
            success_rate = num_completed / (graph_id - train_id_max + 1)
            results = results | {
                "validation_success_rate": success_rate,
                "validation_average_makespan": np.mean(all_makespan),
                "validation_average_partial_success_rate": np.mean(
                    all_partial_success_rate
                ),
                "validation_average_sum_of_costs": np.mean(all_sum_of_costs),
            }

            if args.save_intmd_checkpoints:
                checkpoint_path = pathlib.Path(
                    f"{args.checkpoints_dir}", f"epoch_{epoch}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)

            if success_rate > best_validation_success_rate:
                best_validation_success_rate = success_rate
                if success_rate >= args.threshold_val_success_rate:
                    print("Success rate passed threshold -- Increasing Validation Size")
                    args.threshold_val_success_rate = 1.1
                    cur_validation_id_max = validation_id_max
                    best_val_file_name = "best.pt"
                    best_validation_success_rate = 0.0
                checkpoint_path = pathlib.Path(
                    f"{args.checkpoints_dir}", best_val_file_name
                )
                torch.save(model.state_dict(), checkpoint_path)

            print("Finshed Validation")
            print("------------------")

            if args.run_online_expert and (epoch + 1 >= args.run_oe_after):
                print("---------------------")
                print("Running Online Expert")

                rng = np.random.default_rng(args.dataset_seed + epoch + 1)
                if args.recursive_oe:
                    oe_ids = rng.integers(
                        train_id_max + len(oe_grid_configs), size=args.num_run_oe
                    )
                else:
                    oe_ids = rng.integers(train_id_max, size=args.num_run_oe)

                oe_dataset = []
                oe_hindices = []
                oe_add_data = []
                oe_makespans = []
                num_oe_success = 0
                num_oe_improve = 0

                if oe_improve_quality:
                    aux_func_train(
                        env=None,
                        observations=None,
                        actions=None,
                        oe_period=args.oe_improve_quality_period,
                    )

                for i, graph_id in enumerate(oe_ids):
                    print(f"Running model on {i}/{args.num_run_oe} ", end="")
                    if graph_id > train_id_max:
                        grid_config = oe_grid_configs[graph_id - train_id_max]
                    else:
                        grid_config = _grid_config_generator(
                            seeds[graph_id], map_id=graph_id
                        )
                    success, env, observations = run_model_on_grid(
                        model=model,
                        device=device,
                        grid_config=grid_config,
                        args=args,
                        dataset_kwargs=dataset_kwargs,
                        hypergraph_model=hypergraph_model,
                        max_episodes=args.max_episode_steps,
                        aux_func=aux_func_train if oe_improve_quality else None,
                    )
                    grid_configs_to_run = []
                    if success:
                        num_oe_success += 1
                        if (
                            oe_improve_quality
                            and num_oe_improve < args.oe_improve_quality_max_num
                        ):
                            model_makespan = aux_func_train.makespan
                            expert_makespan = expert_makespans[graph_id]
                            if (
                                model_makespan
                                >= args.oe_improve_quality_buffer * expert_makespan
                            ):
                                # Will run oe to improve solution quality
                                num_oe_improve += 1
                                cur_makespan = model_makespan
                                for gc_to_run in aux_func_train.grid_configs:
                                    cur_makespan -= args.oe_improve_quality_period
                                    # Setting max_episode_steps such that only better solutions are found
                                    gc_to_run.max_episode_steps = int(
                                        cur_makespan / args.oe_improve_quality_buffer
                                    )
                                    grid_configs_to_run.append(gc_to_run)
                    else:
                        grid_configs_to_run.append(generate_grid_config_from_env(env))

                    if len(grid_configs_to_run) > 0:
                        print(f"-- Running OE ", end="")
                        for grid_config in grid_configs_to_run:
                            expert = expert_algorithm(inference_config)

                            all_actions, all_observations, all_terminated = (
                                None,
                                None,
                                None,
                            )
                            expert_results = None
                            hindices = []

                            if args.run_expert_in_separate_fork:
                                done_event.clear()
                                p = mp.Process(
                                    target=multiprocess_run_expert,
                                    args=(
                                        queue,
                                        done_event,
                                        expert,
                                        grid_config,
                                        args.save_termination_state,
                                        hypergraph_generator,
                                    ),
                                )
                                p.start()

                                if args.max_runtime_oe is not None:
                                    start_time = time.time()

                                while p.is_alive():
                                    if args.max_runtime_oe is not None:
                                        if (
                                            time.time() - start_time
                                            > args.max_runtime_oe
                                        ):
                                            print(f"-- Timeout")
                                            p.terminate()
                                            break
                                    try:
                                        expert_results = queue.get(timeout=3)
                                        done_event.set()
                                        p.join(timeout=0.5)
                                        if p.exitcode is None:
                                            p.terminate()
                                        break
                                    except:
                                        p.join(timeout=0.5)
                                        if p.exitcode is not None:
                                            break
                            else:
                                multiprocess_run_expert(
                                    queue,
                                    None,
                                    expert,
                                    grid_config,
                                    args.save_termination_state,
                                    hypergraph_generator,
                                )
                                expert_results = queue.get()

                            if expert_results is not None:
                                (all_actions, all_observations, all_terminated) = (
                                    expert_results[:3]
                                )
                                grid_config = expert_results[-1]
                                if hypergraph_model:
                                    hindices = expert_results[-2]
                                if all(all_terminated[-1]):
                                    print(f"-- Success")
                                    oe_dataset.append(
                                        (all_observations, all_actions, all_terminated)
                                    )
                                    oe_hindices.extend(hindices)
                                    grid_config.max_episode_steps = (
                                        args.max_episode_steps
                                    )
                                    oe_grid_configs.append(grid_config)
                                    oe_makespans.append(len(all_actions))
                                    if load_additional_data:
                                        add_data = generate_additional_data(
                                            grid_config=grid_config,
                                            all_actions=all_actions,
                                            num_previous_actions=args.add_data_num_previous_actions,
                                            cost_to_go=args.add_data_cost_to_go,
                                            normalized_cost_to_go=args.normalize_cost_to_go,
                                            greedy_action=args.add_data_greedy_action,
                                            clamp_value=args.clamp_cost_to_go,
                                            clamped_values_doubled=args.clamped_values_doubled,
                                        )
                                        oe_add_data.extend(add_data)
                                else:
                                    print(f"-- Fail")
                                    break
                            else:
                                print(f"-- Error")
                                break
                    else:
                        print(f"-- Success")
                oe_success_rate = num_oe_success / len(oe_ids)
                if (
                    not oe_improve_quality
                    and args.oe_improve_quality
                    and oe_success_rate >= args.oe_improve_quality_threshold
                ):
                    print("Enabling Online Expert for Solution Quality Improvement")
                    oe_improve_quality = True
                    if args.oe_improve_quality_expert is not None:
                        args.expert_algorithm = args.oe_improve_quality_expert
                        print(f"Switching expert to {args.expert_algorithm}.")
                        expert_algorithm, inference_config = (
                            get_expert_algorithm_and_config(args)
                        )
                while queue.qsize() > 0:
                    # Popping remaining elements, although no elements should remain
                    expert_results = queue.get()
                    hindices = []
                    (all_actions, all_observations, all_terminated) = expert_results[:3]
                    grid_config = expert_results[-1]
                    if hypergraph_model:
                        hindices = expert_results[-2]

                    if all(all_terminated[-1]):
                        oe_dataset.append(
                            (all_observations, all_actions, all_terminated)
                        )
                        oe_hindices.extend(hindices)
                        grid_config.max_episode_steps = args.max_episode_steps
                        oe_grid_configs.append(grid_config)
                        oe_makespans.append(len(all_actions))
                        if load_additional_data:
                            add_data = generate_additional_data(
                                grid_config=grid_config,
                                all_actions=all_actions,
                                num_previous_actions=args.add_data_num_previous_actions,
                                cost_to_go=args.add_data_cost_to_go,
                                normalized_cost_to_go=args.normalize_cost_to_go,
                                greedy_action=args.add_data_greedy_action,
                                clamp_value=args.clamp_cost_to_go,
                                clamped_values_doubled=args.clamped_values_doubled,
                            )
                            oe_add_data.extend(add_data)

                if len(oe_dataset) > 0:
                    print(f"Adding {len(oe_dataset)} OE grids to the dataset")
                    oe_hindices = tuple(list(h) for h in zip(*oe_hindices))
                    for i in range(len(oe_hypergraph_indices)):
                        oe_hypergraph_indices[i].extend(oe_hindices[i])
                    new_oe_graph_dataset = generate_graph_dataset(
                        dataset=oe_dataset,
                        comm_radius=args.comm_radius,
                        obs_radius=args.obs_radius,
                        num_samples=None,
                        save_termination_state=True,
                        use_edge_attr=dataset_kwargs["use_edge_attr"],
                        print_prefix=None,
                        num_neighbour_cutoff=args.num_neighbour_cutoff,
                        neighbour_cutoff_method=args.neighbour_cutoff_method,
                        distance_metric=args.distance_metric,
                        random_edge_probs=args.random_edge_probs,
                        stack_with_np=stack_with_np,
                    )
                    if oe_graph_dataset is None:
                        oe_graph_dataset = new_oe_graph_dataset
                    else:
                        if isinstance(oe_graph_dataset[0], torch.Tensor):
                            oe_graph_dataset = tuple(
                                torch.concat(
                                    [oe_graph_dataset[i], new_oe_graph_dataset[i]],
                                    dim=0,
                                )
                                for i in range(len(oe_graph_dataset))
                            )
                        else:
                            oe_graph_dataset = tuple(
                                oe_graph_dataset[i] + new_oe_graph_dataset[i]
                                for i in range(len(oe_graph_dataset))
                            )
                    if len(oe_add_data) > 0:
                        if oe_additional_data is None:
                            oe_additional_data = oe_add_data
                        else:
                            oe_additional_data.extend(oe_add_data)

                    if expert_makespans is not None:
                        oe_makespans = np.array(oe_makespans)
                        expert_makespans = np.concatenate(
                            [expert_makespans, oe_makespans], axis=0
                        )

                print("Finished Online Expert")
                print("----------------------")

        if use_wandb:
            wandb.log(results)
    checkpoint_path = pathlib.Path(f"{args.checkpoints_dir}", f"last.pt")
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    mp.set_start_method("fork")  # TODO: Maybe add this as an cmd line option
    main()
