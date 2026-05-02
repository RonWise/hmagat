import torch
import torch.nn.functional as F

from torch_geometric.utils import scatter

import math

NUM_CLASSES = 5


class CombinedModel(torch.nn.Module):
    def __init__(self, model, temp_predictor, storage=True):
        super().__init__()
        self.model = model
        self.temp_predictor = temp_predictor
        self.storage = storage
        self.pre_gnn_input = None
        self.saved_logits = None
        self.device = model.device

    def in_simulation(self, in_simulation):
        self.model.in_simulation(in_simulation)

    def forward(self, x, gdata):
        out, pre_gnn_input = self.model(x, gdata, return_pre_gnn_input=True)

        if self.storage:
            self.pre_gnn_input = pre_gnn_input
            self.saved_logits = out

        temperature = self.temp_predictor(out, pre_gnn_input, gdata)
        self.post_temp_logits = out / temperature

        return self.post_temp_logits


class ActorModel(torch.nn.Module):
    def __init__(self, min_value=1e-8, max_value=1.0):
        super().__init__()
        self.clamp_output = False
        self.scale_value = True

        if min_value < 0.1 and max_value > 0.9:
            self.clamp_output = True
            self.scale_value = False

        self.min_value = min_value
        self.max_value = max_value

    def pre_clamp_temperature(self, logits, pre_gnn_input, gdata):
        raise NotImplementedError

    def forward(self, logits, pre_gnn_input, gdata):
        temperature = self.pre_clamp_temperature(logits, pre_gnn_input, gdata)
        temperature = torch.sigmoid(temperature)

        if self.clamp_output:
            temperature = temperature.clamp(self.min_value, self.max_value)

        if self.scale_value:
            temperature = (self.max_value - self.min_value) * temperature
            temperature += self.min_value

        return temperature


class SimpleLocalActor(ActorModel):
    def __init__(self, embedding_size=32, **kwargs):
        super().__init__(**kwargs)
        self.logit_lin = torch.nn.Linear(NUM_CLASSES, embedding_size, bias=False)
        self.target_vec_lin = torch.nn.Linear(3, embedding_size, bias=False)
        self.num_neighbours_lin = torch.nn.Linear(1, embedding_size, bias=False)
        self.num_obstacles_lin = torch.nn.Linear(1, embedding_size, bias=False)

        self.process_lin = torch.nn.Linear(embedding_size, 1)

    def constant_value_init(self, value, scale=1e-3):
        with torch.no_grad():
            # Scaling down the weights
            self.process_lin.weight *= scale

        assert value > self.min_value and value < self.max_value

        if self.scale_value:
            value = (value - self.min_value) / (self.max_value - self.min_value)

        value = 1 / value - 1
        value = 1 / value
        value = math.log(value)

        torch.nn.init.constant_(self.process_lin.bias, value)

    def pre_clamp_temperature(self, logits, pre_gnn_input, gdata):
        num_cells = (gdata.x.shape[1] - 2) * (
            gdata.x.shape[2] - 2
        )  # Accounting for padding

        num_neighbours = torch.sum(gdata.x[:, :, :, 1], dim=(1, 2)) / num_cells
        num_obstacles = torch.sum(gdata.x[:, :, :, 0], dim=(1, 2)) / num_cells

        target_vec = gdata.target_vec[:, :2]
        target_vec_dist = torch.abs(target_vec).sum(dim=-1, keepdim=True)
        target_vec = torch.concatenate([target_vec, target_vec_dist], dim=-1)

        x = self.logit_lin(logits)
        x += self.target_vec_lin(target_vec)
        x += self.num_neighbours_lin(num_neighbours.unsqueeze(-1))
        x += self.num_obstacles_lin(num_obstacles.unsqueeze(-1))

        x = torch.relu(x)
        x = self.process_lin(x)

        return x


class SimpleLocalCritic(torch.nn.Module):
    def __init__(self, embedding_size=32):
        super().__init__()
        self.logit_lin = torch.nn.Linear(NUM_CLASSES, embedding_size, bias=False)
        self.target_vec_lin = torch.nn.Linear(3, embedding_size, bias=False)
        self.num_neighbours_lin = torch.nn.Linear(1, embedding_size, bias=False)
        self.num_obstacles_lin = torch.nn.Linear(1, embedding_size, bias=False)

        self.process_lin = torch.nn.Linear(embedding_size, 1)

    def scale_non_target_lins(self, scale=1e-3):
        with torch.no_grad():
            # Scaling down the weights
            self.logit_lin.weight *= scale
            self.num_neighbours_lin.weight *= scale
            self.num_obstacles_lin.weight *= scale

    def forward(self, logits, pre_gnn_input, gdata, next_values=False):
        num_cells = (gdata.x.shape[1] - 2) * (
            gdata.x.shape[2] - 2
        )  # Accounting for padding

        x = gdata.x
        target_vec = gdata.target_vec[:, :2]
        if next_values:
            x = gdata.next_x
            target_vec = gdata.next_target_vec[:, :2]

        num_neighbours = torch.sum(x[:, :, :, 1], dim=(1, 2)) / num_cells
        num_obstacles = torch.sum(x[:, :, :, 0], dim=(1, 2)) / num_cells

        target_vec_dist = torch.abs(target_vec).sum(dim=-1, keepdim=True)
        target_vec = torch.concatenate([target_vec, target_vec_dist], dim=-1)

        x = self.logit_lin(logits)
        x += self.target_vec_lin(target_vec)
        x += self.num_neighbours_lin(num_neighbours.unsqueeze(-1))
        x += self.num_obstacles_lin(num_obstacles.unsqueeze(-1))

        x = torch.relu(x)
        x = self.process_lin(x)

        # Aggregating x and sharing across all agents
        x = scatter(x, gdata.batch, dim=0, reduce="mean")
        x = x[gdata.batch]

        return x


def clip_ppo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(rewards.size(1))):
        mask = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_values[:, t] * mask - values[:, t]
        advantages[:, t] = last_advantage = delta + gamma * lam * mask * last_advantage

    returns = advantages + values
    return advantages, returns


def compute_log_probs(logits, actions):
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)
    return log_probs


def get_actor_critic(model, args, device):
    actor_kwargs = dict(
        min_value=args.temperature_min_val, max_value=args.temperature_max_val
    )
    if args.temperature_actor_critic == "simple-local":
        actor = SimpleLocalActor(
            embedding_size=args.temperature_embedding_size, **actor_kwargs
        ).to(device)
        critic = SimpleLocalCritic(embedding_size=args.temperature_embedding_size).to(
            device
        )
    elif args.temperature_actor_critic == "simple-local-val-init":
        actor = SimpleLocalActor(
            embedding_size=args.temperature_embedding_size, **actor_kwargs
        ).to(device)
        actor.constant_value_init(value=0.75)
        critic = SimpleLocalCritic(embedding_size=args.temperature_embedding_size).to(
            device
        )
        critic.scale_non_target_lins()
    else:
        raise ValueError(
            f"Unknown temperature actor-critic type: {args.temperature_actor_critic}."
        )
    return actor, critic
