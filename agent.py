import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_scale: int):
        super(QNetwork, self).__init__()
        # shared state feature extraction layer
        self.feature = nn.Sequential(nn.Linear(state_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU()
                                     )
        # evaluate action advantages on each branch
        self.actions = [nn.Sequential(nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, action_scale)
                                      ) for _ in range(action_dim)]
        self.actions = nn.ModuleList(self.actions)
        # module to calculate state value
        self.value = nn.Sequential(nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 1)
                                   )

    def forward(self, x):
        feature = self.feature(x)
        actions = torch.stack([head(feature) for head in self.actions])
        value = self.value(feature)
        maxa = actions.mean(-1).max(0)[0].unsqueeze(-1)
        actions = actions - maxa + value
        # maxmax = actions.max(0)[0].max(-1)[0].unsqueeze(-1)
        # actions = actions - maxmax + value
        return actions


class BDQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_scale: int, learning_rate, device: str):
        super(BDQ, self).__init__()

        self.q = QNetwork(state_dim, action_dim, action_scale).to(device)
        self.target_q = QNetwork(state_dim, action_dim, action_scale).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam([{'params': self.q.feature.parameters(), 'lr': learning_rate / action_dim},
                                     {'params': self.q.value.parameters(), 'lr': learning_rate / action_dim},
                                     {'params': self.q.actions.parameters(), 'lr': learning_rate}])
        self.update_freq = 1000
        self.update_count = 0

    def take_action(self, x):
        return self.q(x)
    
    def append_sample(self, memory, state, action, reward, next_state, done_mask):
        memory.add((state, action, reward, next_state, done_mask))

    def update(self, memory, batch_size, gamma):
        state, actions, reward, next_state, done_mask = memory.sample(batch_size)
        actions = torch.stack(actions).transpose(0, 1).unsqueeze(-1)
        done_mask = torch.abs(done_mask - 1)

        q_values = self.q(state).transpose(0, 1)  # q_values for all possible actions
        q_values = q_values.gather(2, actions.long()).squeeze(-1)  # get q_values for current action

        # select best actions from Q and calculate Q-values in target Q
        max_next_action = self.q(next_state).transpose(0, 1).max(-1, keepdim=True)[1]
        max_q_values = self.target_q(next_state).transpose(0, 1)  # normal dqn
        max_next_q_values = max_q_values.gather(2, max_next_action.long()).squeeze(-1)  # get q_values for next action
        # max_next_q_values = max_q_values.max(-1, keepdim=True)[0].squeeze(-1)
        q_target = (done_mask * gamma * max_next_q_values + reward)

        loss = F.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.update_freq == 0:
            self.update_count = 0
            self.target_q.load_state_dict(self.q.state_dict())

        return loss
