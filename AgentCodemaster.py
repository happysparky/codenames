import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
from Codemaster import Codemaster

DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_NUMBER_OF_APPLICABLE_WORDS = 9

'''
can probably rename this class to AgentCodemaster or something to follow same pattern as humanCodemaster
'''

class AgentCodemaster(Codemaster):
    def __init__(self, params, i2v):
        super().__init__()

        self.i2v = i2v

        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 0.1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()
          
    def network(self):
        # Layers
        self.f1 = nn.Linear(6730, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.hintLin = nn.Linear(self.third_layer, len(self.i2v))
        self.countLin = nn.Linear(self.third_layer, MAX_NUMBER_OF_APPLICABLE_WORDS)
        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        hintTensor = F.softmax(self.hintLin(x), dim=-1)
        countTensor = F.softmax(self.countLin(x), dim=-1)
        return hintTensor, countTensor

    def set_reward(self, num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed, game_ended):
        ''' I think the danger word penality needs to increase. Max an agent can get is +80 or +90 
        if they guessed all 8 or 9 words in one go. Guessing the danger word should outweigh up to one step below'''
        """
        Return the reward.
        The reward is:
            -50 when Player guesses danger word. 
            -10 when Player guesses opposing-team word. 
            +10 when Player guesses own-team food
            -5 when Player guesses neutral word
        """

        # TODO: add "don't-suggest-a-previous-hint" penalty
        # TODO: don't give more a clue that applies to more words than there are left
        # TODO: add "don't suggest a number greater than the words remaining"

        self.reward = 10*num_own_guessed - 10*num_opposing_guessed - 5*num_neutral_guessed - 50*num_danger_guessed
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = (reward, reward)
            next_state_tensor = torch.flatten(torch.tensor(np.expand_dims(next_state, 0), dtype=torch.double)).to(DEVICE)
            state_tensor = torch.flatten(torch.tensor(np.expand_dims(state, 0), dtype=torch.double, requires_grad=True)).to(DEVICE)
            if not done:
                hint_target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
                count_target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[1])
                target = (hint_target, count_target)
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target[0]
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()            


    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = (reward, reward)

        # a state contains 10 things to consider
        # each of the 7 things has 673 (vocab size) spots
        # 10 x 673 = 6730
        next_state_tensor = next_state.double().clone().detach().reshape((1, 6730)).to(DEVICE)
        state_tensor = state.double().clone().detach().reshape((1, 6730)).requires_grad_(True).to(DEVICE)


        if not done:
            hint_target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            count_target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[1])
            target = (hint_target, count_target)
        hintOutput, countOutput = self.forward(state_tensor)
        target_f_hint, target_f_count = hintOutput.clone(), countOutput.clone()
        
        # Create targets for both hint and count
        # Note: may want to validate that these look correct
        # print("target", target_f_hint, target_f_count, '\n')
        # print("hint argmax & hint value", torch.argmax(action), torch.max(action), '\n')
        # print("target_f", target_f_hint[0], target_f_count[0], '\n')

        target_f_hint[0][torch.argmax(action)] = target[0]
        target_f_hint.detach()
        target_f_count[0][math.floor(torch.max(action))] = target[1]
        target_f_count.detach()

        self.optimizer.zero_grad()

        # Sum loss for both hint and count
        loss = F.mse_loss(hintOutput, target_f_hint)
        loss += F.mse_loss(countOutput, target_f_count)

        loss.backward()
        self.optimizer.step()