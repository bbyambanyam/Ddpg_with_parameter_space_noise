import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, state_dimension, action_dimension, action_max):
        super(Actor, self).__init__()

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.action_max = action_max

        # 3 layer uusgene

        self.fc1 = nn.Linear(state_dimension, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dimension)
        
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        output = self.fc1(state)
        output = F.relu(output)

        output = self.fc2(output)
        output = F.relu(output)

        output = self.fc3(output)
        output = F.relu(output)

        action = self.fc4(output)
        action = F.tanh(action)

        action = action * self.action_max

        # action butsaana

        return action

class Critic(nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super(Critic, self).__init__()

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        
        # 4 layer uusgene

        self.fcs1 = nn.Linear(state_dimension, 256)
        self.fcs2 = nn.Linear(256, 128)
        self.fca1 = nn.Linear(action_dimension, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        s1 = self.fcs1(state)
        s1 = F.relu(s1)
        s2 = self.fcs2(s1)
        s2 = F.relu(s2)
        a1 = self.fca1(action)
        a1 = F.relu(a1)

        output = torch.cat((s2, a1), dim=1)
        output = self.fc2(output)
        output = F.relu(output)
        q_value = self.fc3(output)

        # Q value-g butsaana

        return q_value