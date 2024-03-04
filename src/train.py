from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
path='model.pt'
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)
    
class ProjectAgent:
    def act(self, observation, use_random=False):
        Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
        return Q.argmax().item()

    def save(self):
        path='model.pt'
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load('model.pt', map_location=device))
        self.model.eval()

    def __init__(self):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'],device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.total_steps = 0
        self.model = model 
        if config['use_Huber_loss']:
            self.criterion = torch.nn.SmoothL1Loss()
        else:
            self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps']
        self.target_model = deepcopy(self.model).to(device)
        self.update_target_strategy = config['update_target_strategy']
        self.update_target_freq = config['update_target_freq']
        self.update_target_tau = config['update_target_tau']
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode=200):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score = -1
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1*tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                score = 0
                if episode > 100:
                    score = evaluate_HIV(agent=self, nb_episode=1)
                
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      ", score ", '{:.2e}'.format(score),
                      sep='')
                state, _ = env.reset()

                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
                
                if score > best_score:
                    best_score = score
                    self.save()

            else:
                state = next_state

            
        return episode_return
'''
    if __name__ == "__main__":
         config = {'nb_actions': env.action_space.n,
              'learning_rate': 0.001,
              'gamma': 0.95,
              'buffer_size': 100000,
              'epsilon_min': 0.02,
              'epsilon_max': 1.,
              'epsilon_decay_period': 20000,
              'epsilon_delay_decay': 500,
              'gradient_steps': 3,
              'update_target_strategy': 'replace',
              'update_target_freq': 400,
              'update_target_tau': 0.005,
              'use_Huber_loss': True,
              'batch_size': 800,
              }
         model = DQN(env.observation_space.shape[0],env.action_space.n)
         print(env)
         ProjectAgent.train(env, 200)
'''
config = {'nb_actions': env.action_space.n,
              'learning_rate': 0.001,
              'gamma': 0.95,
              'buffer_size': 100000,
              'epsilon_min': 0.02,
              'epsilon_max': 1.,
              'epsilon_decay_period': 20000,
              'epsilon_delay_decay': 500,
              'gradient_steps': 3,
              'update_target_strategy': 'replace',
              'update_target_freq': 400,
              'update_target_tau': 0.005,
              'use_Huber_loss': True,
              'batch_size': 800,
              }
model = DQN(env.observation_space.shape[0],env.action_space.n)
agent = ProjectAgent()
#ep_length = agent.train(env, 200)