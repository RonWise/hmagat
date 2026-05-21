import argparse
import pathlib
import numpy as np
import wandb
from collections import OrderedDict

from pogema import pogema_v0

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Batch

from hmagat.run_expert import add_expert_dataset_args

from hmagat.training_args import add_training_args, validate_training_args_contract
from hmagat.convert_to_imitation_dataset import add_imitation_dataset_args
from hmagat.generate_hypergraphs import add_hypergraph_generation_args

# from agents import run_model_on_grid, get_model
from hmagat.modules.model.run_model import run_model_on_grid
from grid_config_generator import grid_config_generator_factory

from hmagat.generate_additional_data import add_additional_data_args
from hmagat.checkpointing import (
    load_model_state_dict_from_checkpoint_path,
    resolve_evaluation_checkpoint_path,
)

from hmagat.modules.temperature_sampling.actor_critic import (
    get_actor_critic,
    clip_ppo_loss,
    compute_gae,
    compute_log_probs,
    CombinedModel,
)

from hmagat.runtime_data_generation import get_runtime_data_generator
from hmagat.collision_shielding import get_collision_shielded_model


def add_temperature_sampling_args(parser):
    parser.add_argument(
        "--rl_based_temperature_sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--temperature_actor_critic", type=str, default="simple")
    parser.add_argument("--temperature_rl_gamma", type=float, default=0.99)
    parser.add_argument("--temperature_rl_lam", type=float, default=0.95)
    parser.add_argument("--temperature_rl_clip_epsilon", type=float, default=0.2)
    parser.add_argument("--temperature_rl_lr", type=float, default=3e-4)
    parser.add_argument(
        "--temperature_checkpoints_dir", type=str, default="temp_checkpoints"
    )
    parser.add_argument("--temperature_optimize", type=str, default=None)
    parser.add_argument("--temperature_embedding_size", type=int, default=32)

    parser.add_argument("--iterations_per_epoch", type=int, default=1)
    parser.add_argument("--temperature_run_name", type=str, default="simple")
    parser.add_argument(
        "--temperature_run_for_baseline_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--temperature_min_val", type=float, default=1e-8)
    parser.add_argument("--temperature_max_val", type=float, default=1.0)

    return parser


def create_data(
    gdata,
    actions,
    rewards,
    terminated,
    truncated,
    saved_logits,
    pre_gnn_input,
    old_log_probs,
    next_gdata,
    next_logits,
    next_gnn_input,
):
    gdata = gdata.cpu()
    actions = actions.cpu()
    rewards = torch.tensor(rewards, dtype=torch.float32).cpu()

    terminated = torch.tensor(terminated, dtype=torch.float32).cpu()
    truncated = torch.tensor(truncated, dtype=torch.float32).cpu()
    terminated = torch.clip(
        terminated + truncated, 0, 1
    )  # Combine terminated and truncated

    saved_logits = saved_logits.cpu()
    pre_gnn_input = pre_gnn_input.cpu()
    old_log_probs = old_log_probs.cpu()
    next_gdata = next_gdata.cpu()
    next_logits = next_logits.cpu()
    next_gnn_input = next_gnn_input.cpu()

    return Data(
        x=gdata.x,
        edge_index=gdata.edge_index,
        edge_attr=gdata.edge_attr,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        saved_logits=saved_logits,
        pre_gnn_input=pre_gnn_input,
        old_log_probs=old_log_probs,
        target_vec=gdata.target_vec,
        next_x=next_gdata.x,
        next_edge_index=next_gdata.edge_index,
        next_edge_attr=next_gdata.edge_attr,
        next_target_vec=next_gdata.target_vec,
        next_logits=next_logits,
        next_gnn_input=next_gnn_input,
    )

def only_all_on_goal_reward(
    env,
    observations,
    previous_observations,
    info,
    actions,
    args,
    rewards,
    reward_calculation_state=None,
):
    if reward_calculation_state is None:
        reward_calculation_state = {
            "prev_at_goals": np.zeros(env.num_agents, dtype=bool)
        }

    at_goals = np.array(env.was_on_goal)
    if at_goals.all():
        # All agents are at goals, return +1 reward
        rewards = [1.0] * env.num_agents
    else:
        # -1 regardless of location
        rewards = [-1.0] * env.num_agents

    reward_calculation_state = {"prev_at_goals": at_goals}
    return rewards, reward_calculation_state

def calculate_rewards(
    env,
    observations,
    previous_observations,
    info,
    actions,
    args,
    rewards,
    reward_calculation_state=None,
):
    if args.temperature_optimize == "only-all-on-goal":
        return only_all_on_goal_reward(
            env=env,
            observations=observations,
            previous_observations=previous_observations,
            info=info,
            actions=actions,
            args=args,
            rewards=rewards,
            reward_calculation_state=reward_calculation_state,
        )
    else:
        raise NotImplementedError(
            f"Reward calculation for {args.temperature_optimize} is not implemented."
        )


@torch.no_grad()
def run_model_and_collect_data(
    model,
    device,
    grid_config,
    args,
    hypergraph_model,
    dataset_kwargs,
    use_target_vec,
):
    datas = []
    env = pogema_v0(grid_config=grid_config)
    observations, infos = env.reset()

    rt_data_generator = get_runtime_data_generator(
        grid_config=grid_config,
        args=args,
        hypergraph_model=hypergraph_model,
        dataset_kwargs=dataset_kwargs,
        use_target_vec=use_target_vec,
    )

    reward_calculation_state = None
    cs_model = get_collision_shielded_model(model, env, args)
    while True:
        gdata = rt_data_generator(observations, env).to(device)
        actions = cs_model.get_actions(gdata)
        prev_observations = observations
        observations, rewards, terminated, truncated, infos = env.step(actions)

        if args.temperature_optimize is not None:
            # Updating rewards
            rewards, reward_calculation_state = calculate_rewards(
                env=env,
                observations=observations,
                previous_observations=prev_observations,
                info=infos,
                actions=actions,
                args=args,
                rewards=rewards,
                reward_calculation_state=reward_calculation_state,
            )

        actions = torch.from_numpy(actions).to(device)
        old_log_probs = compute_log_probs(model.post_temp_logits, actions)
        datas.append(
            (
                gdata,
                actions,
                rewards,
                terminated,
                truncated,
                model.saved_logits,
                model.pre_gnn_input,
                old_log_probs,
            )
        )

        if all(terminated) or all(truncated):
            break
    # Adding next state
    gdata = rt_data_generator(observations, env).to(device)
    actions = cs_model.get_actions(gdata)
    pre_gnn_input = model.pre_gnn_input
    saved_logits = model.saved_logits

    datas.append((gdata, None, None, None, None, saved_logits, pre_gnn_input, None))

    gdatas = []
    prev_data = datas[0]
    for data in datas[1:]:
        next_gdata, _, _, _, _, next_logits, next_gnn_input, _ = data
        gdatas.append(create_data(*prev_data, next_gdata, next_logits, next_gnn_input))
        prev_data = data

    return gdatas, infos, all(terminated)


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


def get_temperature_sampling_model(model, args, device, state_dict=None):
    actor, _ = get_actor_critic(model, args, device)
    actor = actor.eval()

    if state_dict is not None:
        actor.load_state_dict(state_dict)

    combined_model = CombinedModel(model, actor).to(device)

    return combined_model


def main():
    parser = argparse.ArgumentParser(description="Train temperature sampling.")
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_additional_data_args(parser)
    parser = add_training_args(parser)
    parser = add_temperature_sampling_args(parser)

    parser.add_argument("--model_epoch_num", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default="hyper-mapf-temp-samp")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    validate_training_args_contract(args)
    print(args)

    assert args.save_termination_state
    assert args.rl_based_temperature_sampling or args.temperature_run_for_baseline_only

    if args.device == -1:
        device = torch.device("cuda")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    rng = np.random.default_rng(args.dataset_seed)
    seeds = rng.integers(10**10, size=args.num_samples)

    _grid_config_generator = grid_config_generator_factory(args)

    from hmagat.modules.agents import get_model

    model, hypergraph_model, dataset_kwargs = get_model(args, device)

    checkpoint_path = resolve_evaluation_checkpoint_path(
        args.checkpoints_dir,
        model_epoch_num=args.model_epoch_num,
        map_location=device,
    )

    state_dict = load_model_state_dict_from_checkpoint_path(
        checkpoint_path,
        map_location=device,
        print_prefix="[temperature-base-load] ",
    )
    model.load_state_dict(state_dict)

    model = model.eval()
    model.in_simulation(True)

    train_id_max = int(args.num_samples * (1 - args.validation_fraction))

    org_results = dict()

    if not args.skip_validation:
        print("Running original validation...")
        num_completed = 0
        all_makespan = []
        all_partial_success_rate = []
        all_sum_of_costs = []

        for seed_id in range(train_id_max, args.num_samples):
            success, env, observations = run_model_on_grid(
                model=model,
                device=device,
                grid_config=_grid_config_generator(seeds[seed_id]),
                args=args,
                dataset_kwargs=dataset_kwargs,
                hypergraph_model=hypergraph_model,
                use_target_vec=args.use_target_vec,
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
        success_rate = num_completed / (args.num_samples - train_id_max)
        org_results = {
            "org_validation_success_rate": success_rate,
            "org_validation_makespan": np.mean(all_makespan),
            "org_validation_partial_success_rate": np.mean(all_partial_success_rate),
            "org_validation_sum_of_costs": np.mean(all_sum_of_costs),
        }
    use_wandb = args.wandb_entity is not None
    assert use_wandb, "Using wandb to log information"

    if args.temperature_run_for_baseline_only:
        run_name = f"{args.run_name}_{args.temperature_run_name}"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            entity=args.wandb_entity,
        )
        for _ in range(args.num_epochs):
            wandb.log(org_results)
        return

    actor, critic = get_actor_critic(model, args, device)
    actor = actor.eval()
    critic = critic.eval()
    combined_model = CombinedModel(model, actor).to(device)

    # Hyperparameters for PPO
    gamma = args.temperature_rl_gamma
    lam = args.temperature_rl_lam
    clip_epsilon = args.temperature_rl_clip_epsilon
    lr = args.temperature_rl_lr

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    checkpoint_path = pathlib.Path(args.temperature_checkpoints_dir, "last.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = f"{args.run_name}_{args.temperature_run_name}"
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            entity=args.wandb_entity,
        )

    use_target_vec = args.use_target_vec
    if use_target_vec is None:
        # Setting use_target_vec so that actors and critics can use them
        use_target_vec = "target-vec"

    print("Starting Training....")
    for epoch in range(args.num_epochs):
        dataset = []
        run_makespan = []
        run_success = 0
        run_soc = []
        mean_actor_loss = 0.0
        mean_critic_loss = 0.0
        mean_rewards = 0.0
        for i in range(train_id_max):
            grid_config = _grid_config_generator(seeds[i])

            print(f"Collecting data for sample {i + 1}/{train_id_max}...", end="\r")
            datas, infos, success = run_model_and_collect_data(
                combined_model,
                device,
                grid_config,
                args,
                hypergraph_model,
                dataset_kwargs,
                use_target_vec=use_target_vec,
            )
            dataset.append(datas)

            run_makespan.append(infos[0]['metrics']['makespan'])
            run_soc.append(infos[0]['metrics']['SoC'])

            if success:
                run_success += 1

        for _ in range(args.iterations_per_epoch):
            for data in dataset:
                n_agents = data[0].num_nodes
                data = Batch.from_data_list(data)
                data = data.to(device)

                # Get values and next values
                values = critic(data.saved_logits, data.pre_gnn_input, data)
                next_values = critic(
                    data.next_logits, data.next_gnn_input, data, next_values=True
                )

                # Compute advantages and returns
                advantages, returns = compute_gae(
                    data.rewards.reshape((n_agents, -1)),
                    values.reshape((n_agents, -1)),
                    next_values.reshape((n_agents, -1)),
                    data.terminated.reshape((n_agents, -1)),
                    gamma=gamma,
                    lam=lam,
                )

                advantages = advantages.reshape(-1)
                returns = returns.reshape(-1)

                mean_rewards += data.rewards.mean().item()

                # Compute new log probabilities
                temperature = actor(data.saved_logits, data.pre_gnn_input, data)
                new_logits = data.saved_logits / temperature
                new_log_probs = compute_log_probs(new_logits, data.actions)

                # Compute PPO loss
                ppo_loss = clip_ppo_loss(
                    data.old_log_probs,
                    new_log_probs,
                    advantages,
                    clip_epsilon=clip_epsilon,
                )
                mean_actor_loss += ppo_loss.item()

                # Update actor
                actor_optimizer.zero_grad()
                ppo_loss.backward(retain_graph=True)
                actor_optimizer.step()

                # Update critic
                critic_optimizer.zero_grad()
                value_loss = F.mse_loss(values.squeeze(), returns.squeeze())
                mean_critic_loss += value_loss.item()
                value_loss.backward()
                critic_optimizer.step()

        mean_actor_loss /= len(dataset)
        mean_critic_loss /= len(dataset)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs}: Mean Actor Loss: {mean_actor_loss:.4f}, Mean Critic Loss: {mean_critic_loss:.4f}"
        )

        results = {
            "mean_actor_loss": mean_actor_loss,
            "mean_critic_loss": mean_critic_loss,
            "mean_rewards": mean_rewards / len(dataset) / args.iterations_per_epoch,
            "run_success_rate": run_success / len(dataset),
            "run_makespan": np.mean(run_makespan),
            "run_soc": np.mean(run_soc),
        }

        if (not args.skip_validation) and (
            (epoch + 1) % args.validation_every_epochs == 0
        ):
            print("Running validation...")
            num_completed = 0
            all_makespan = []
            all_partial_success_rate = []
            all_sum_of_costs = []

            for seed_id in range(train_id_max, args.num_samples):
                success, env, observations = run_model_on_grid(
                    model=combined_model,
                    device=device,
                    grid_config=_grid_config_generator(seeds[seed_id]),
                    args=args,
                    dataset_kwargs=dataset_kwargs,
                    hypergraph_model=hypergraph_model,
                    use_target_vec=use_target_vec,
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
            success_rate = num_completed / (args.num_samples - train_id_max)
            results = results | {
                "validation_success_rate": success_rate,
                "validation_makespan": np.mean(all_makespan),
                "validation_partial_success_rate": np.mean(all_partial_success_rate),
                "validation_sum_of_costs": np.mean(all_sum_of_costs),
            }

            checkpoint_path = pathlib.Path(
                args.temperature_checkpoints_dir, f"epoch_{epoch}.pt"
            )
            torch.save(actor.state_dict(), checkpoint_path)

        if use_wandb:
            wandb.log(results | org_results)
    checkpoint_path = pathlib.Path(args.temperature_checkpoints_dir, "last.pt")
    torch.save(actor.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
