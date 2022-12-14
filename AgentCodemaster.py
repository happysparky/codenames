import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gensim.downloader
from scipy.spatial.distance import cosine
from Codemaster import Codemaster

DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_NUMBER_OF_APPLICABLE_WORDS = 9

class AgentCodemaster(Codemaster):
    def __init__(self, params, i2v, team):
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
        if team == 0:
            self.weights = params['red_codemaster_weights']
        elif team == 1:
            self.weights = params['blue_codemaster_weights']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()
        self.w2v = gensim.downloader.load("glove-wiki-gigaword-50")
          
    def network(self):
        # Layers
        self.f1 = nn.Linear(6720, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.hintLin = nn.Linear(self.third_layer, len(self.i2v))
        self.countLin = nn.Linear(self.third_layer, MAX_NUMBER_OF_APPLICABLE_WORDS)
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

    def set_reward(self, num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, own_team_won=None):
        """
        Return the reward.
        The reward is:
            -50 when Player guesses danger word. 
            -10 when Player guesses opposing-team word. 
            +10 when Player guesses own-team food
            -5 when Player guesses neutral word
        """


        self.reward = 10*num_own_guessed - 10*num_opposing_guessed - 5*num_neutral_guessed - 100*num_danger_guessed
        
        if own_team_won != None:
            if own_team_won == 0:
                self.reward -= 50
            else:
                self.reward += 500

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

        
            hintOutput, countOutput = self.forward(state_tensor) 
            target_f_hint, target_f_count = hintOutput.clone(), countOutput.clone()
            target_f_hint[torch.argmax(action)] = target[0]
            target_f_hint.detach()
            target_f_count[math.floor(torch.max(action))] = target[1]
            target_f_count.detach()

            self.optimizer.zero_grad()

            # Sum loss for both hint and count
            loss = F.mse_loss(hintOutput, target_f_hint)
            loss += F.mse_loss(countOutput, target_f_count)

            loss.backward()
            self.optimizer.step()
            return loss     


    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = (reward, reward)

        # a state contains 10 things to consider
        # each of the 7 things has 672 (vocab size) spots
        # 10 x 672 = 6720
        next_state_tensor = next_state.double().clone().detach().reshape((1, 6720)).to(DEVICE)
        state_tensor = state.double().clone().detach().reshape((1, 6720)).requires_grad_(True).to(DEVICE)


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
        return loss

    # Generates a random hint based on the vocabulary
    def generate_random_hint(self, vocab_size, board, turn, red_words_remaining_count, blue_words_remaining_count):
        hint = random.randint(0, vocab_size-1)
        # ensure hint isn't in list of words
        while hint in board:
            hint = random.randint(0, vocab_size-1)

        words_remaining = red_words_remaining_count if turn == 0 else blue_words_remaining_count
        num_words = random.choice(range(1, words_remaining+1))

        return hint, num_words

    # Generates a hint based on the available vocabulary: min cosine distance to any vocab word not in board
    def generate_structured_hint(self, vocab_size, board, target_word_list):

        distances = []

        base_word = random.choice(target_word_list)
        for i in range(vocab_size):
            guessWord = self.i2v[i]
            if base_word in self.w2v and guessWord in self.w2v and i not in board:
                distances.append(cosine(self.w2v[base_word], self.w2v[guessWord]))
            else:
                distances.append(2)

        hint = np.argmin(distances)

        num_words = random.choice(range(1, len(target_word_list)+1))

        return hint, num_words