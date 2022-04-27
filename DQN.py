import math
from collections import deque, namedtuple
import random
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
environment = 'LunarLander-v2'
env = gym.make(environment)
n_actions = 4
print(env.observation_space.shape)
state_size = 8
max_steps = 1500000
BATCH_SIZE = 32
GAMMA = .99
EPS_START = 0.999
EPS_END = 0.01
EPS_DECAY = 150000
TARGET_UPDATE = 10000
average_size = 10

steps_done = 0
episode_rewards = []
avg_episode_rewards = []
episode_updates = []

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(avg_episode_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    plt.plot(episode_updates, durations_t.numpy())
    # Take X episode averages and plot them too
    # if len(durations_t) >= average_size:
    #     means = durations_t.unfold(0, average_size, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(19), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def select_action(state, model, exp=True):
    global steps_done
    sample = random.random()
    if exp:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
        if .1 > eps_threshold > .09:
            print("less than .1")

    else:
        eps_threshold = 0
    steps_done += 1
    # print(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            x = model(state)
            x = torch.argmax(x)
            x = x.view(1, 1)
            return x
    else:
        # print(eps_threshold)
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_batch = state_batch.view(BATCH_SIZE, state_size)
    state_action_values = policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    non_final_next_states = non_final_next_states.view(BATCH_SIZE, state_size)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    target_output = target(non_final_next_states)
    next_state_values[non_final_mask] = target_output.max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define networks
policy = DQN().to(device)
target = DQN().to(device)
final = DQN().to(device)

target.load_state_dict(policy.state_dict())
target.eval()
final.load_state_dict(policy.state_dict())
final.eval()

optimizer = optim.RMSprop(policy.parameters(), lr=0.0001)
memory = ReplayMemory(200000)




env.reset()

updated_target = True


average_total_reward = 1
max_average_total_reward = 0
i_episode = 0
while steps_done < max_steps:
    # Initialize the environment and state
    total_reward = 0
    state = env.reset()
    state = torch.from_numpy(state)
    for t in count():
        # Select and perform an action
        action = select_action(state, policy)

        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        total_reward += reward

        # Store the transition in memory
        next_state = torch.from_numpy(next_state)
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        state = next_state
        if done:
            average_total_reward += total_reward
            updated_target = False
            break
        # Update the target network, copying all weights and biases in DQN
        if steps_done % 1000 == 0:
            print(steps_done, "Steps done.")
        if steps_done % TARGET_UPDATE == 0:
            print("Target updated.", i_episode)
            target.load_state_dict(policy.state_dict())
            updated_target = True


    if i_episode % average_size == 0:
        plot_durations()
        average_total_reward /= average_size

        avg_episode_rewards.append(average_total_reward)
        episode_updates.append(i_episode)
        if average_total_reward > max_average_total_reward:
            max_average_total_reward = average_total_reward
            final.load_state_dict(policy.state_dict())
        average_total_reward = 1
    i_episode += 1

print('Complete')

while True:
    total_reward = 0
    state = env.reset()
    state = torch.from_numpy(state)
    for t in count():
        env.render()
        action = select_action(state, final, False)
        state, reward, done, _ = env.step(action.item())
        state = torch.from_numpy(state)
        total_reward += reward
        if done:
            print(total_reward)
            break
env.render()
env.close()
plt.ioff()
plt.show()

env.close()
