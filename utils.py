import torch
import numpy as np
import collections
import random


class ReplayBuffer:
    def __init__(self, action_dim, device):
        self.buffer = collections.deque(maxlen=100000)
        self.action_dim = action_dim
        self.device = device

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        actions_lst = [[] for _ in range(self.action_dim)]

        for transition in mini_batch:
            state, actions, reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(self.action_dim):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        
        state_lst = torch.tensor(np.array(state_lst), dtype=torch.float).to(self.device)
        actions_lst = [torch.tensor(x, dtype=torch.float).to(self.device) for x in actions_lst]
        reward_lst = torch.tensor(np.array(reward_lst), dtype=torch.float).to(self.device)
        next_state_lst = torch.tensor(np.array(next_state_lst), dtype=torch.float).to(self.device)
        done_mask_lst = torch.tensor(np.array(done_mask_lst), dtype=torch.float).to(self.device)

        return state_lst, actions_lst, reward_lst, next_state_lst, done_mask_lst

    def size(self):
        return len(self.buffer)
    
