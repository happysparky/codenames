import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

'''
can probably rename this class to AgentCodemaster or something to follow same pattern as humanCodemaster
'''

class Codemaster(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reward = 0
        self.gamma = 0
        self.dataframe = None
        self.short_memory = None
        self.agent_target = 0
        self.agent_predict = 0
        self.learning_rate = None     
        self.epsilon = 0
        self.actual = None
        self.first_layer = None
        self.second_layer = None
        self.third_layer = None
        self.memory = None
        self.weights = None
        self.load_weights = None
        self.optimizer = None
          
    def network(self):
        # Layers
        return

    def forward(self, x):
       return

    def set_reward(self, num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, own_team_won=None):
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

        return self.reward

    def remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        pass

    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        return -1

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        return -1