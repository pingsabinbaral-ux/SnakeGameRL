# import required libraries
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# neural network for deep q learning
class QNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # save trained model weights
    def save(self, name='model.pth'):
        os.makedirs('./model', exist_ok=True)
        torch.save(self.state_dict(), f'./model/{name}')

    # load saved weights if checkpoint exists
    def load(self, name='model.pth'):
        path = f'./model/{name}'
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))
            self.eval()
            return True
        return False


# double dqn trainer
class QTrainer:
    def __init__(self, model, lr, gamma, target_update_freq=1000):
        self.gamma = gamma
        self.model = model
        self.target = copy.deepcopy(model)
        self.target.eval()
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # convert inputs to tensors
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        done = torch.tensor(np.array(done), dtype=torch.float)

        # add batch dimension for single samples
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # compute current q values
        pred = self.model(state)

        # compute target q values using double dqn
        with torch.no_grad():
            best_actions = self.model(next_state).argmax(1, keepdim=True)
            q_next = self.target(next_state).gather(1, best_actions).squeeze(1)
            target_q = reward + self.gamma * q_next * (1 - done)

        # compute loss and run optimizer step
        pred_q = pred.gather(1, action.argmax(1, keepdim=True)).squeeze(1)
        loss = self.criterion(pred_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        # sync target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())
