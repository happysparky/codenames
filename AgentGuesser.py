import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim.downloader
from scipy.spatial.distance import cosine
from Guesser import Guesser

DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

'''
can probably rename this class to AgentGuesser or something to match HumanGuesser
'''

class AgentGuesser(Guesser):
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
            self.weights = params['red_guesser_weights']
        elif team == 1:
            self.weights = params['blue_guesser_weights']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()
        self.w2v = gensim.downloader.load("glove-wiki-gigaword-50")
          
    def network(self):
        # Layers
        self.f1 = nn.Linear(4704, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, len(self.i2v))
        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        # print("GUESSER OUTPUT", x)
        # print(x.size())
        return x

    '''
    NOTE: later, we will try to improve this reward policy. One strategy would be to pass in the two teams' remaining word count. 
    This difference (for example) should motivate a "trailing" team to be riskier. Also add a huge reward for winning the game. 
    '''
    def set_reward(self, num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_prev_guessed, game_ended):
        """
        Return the reward.
        The reward is:
            -50 when Player guesses danger word. 
            -10 when Player guesses opposing-team word. 
            +10 when Player guesses own-team food
            -5 when Player guesses neutral word
            -100 when Player guesses a previously guessed word
        """


        
        self.reward = 10*num_own_guessed - 10*num_opposing_guessed - 5*num_neutral_guessed - 50*num_danger_guessed - 100*num_prev_guessed
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
            target = reward
            # changed float to double
            next_state_tensor = torch.flatten(torch.tensor(np.expand_dims(next_state, 0), dtype=torch.double)).to(DEVICE)
            state_tensor = torch.flatten(torch.tensor(np.expand_dims(state, 0), dtype=torch.double, requires_grad=True)).to(DEVICE)

            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            output = self.forward(state_tensor)
            target_f = output.clone()

            target_f[np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
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
        target = reward

        # a state contains 7 things to consider
        # each of the 7 things has 672 (vocab size) spots
        # 7 x 672 = 4704

        next_state_tensor = next_state.double().clone().detach().reshape((1, 4704)).to(DEVICE)
        state_tensor = state.double().clone().detach().reshape((1, 4704)).requires_grad_(True).to(DEVICE)

        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()
        return loss

    # Guess a random word from the board
    # Performs iterative guessing from all words on the board, checking to make sure that the word hasn't been guessed
    # Also makes sure that not all words have been guessed
    def generate_random_guess(self, red_words_remaining, blue_words_remaining, neutral_words_remaining, danger_words_remaining):
        possible_guesses = np.concatenate(
            (
                np.nonzero(red_words_remaining)[0],
                np.nonzero(blue_words_remaining)[0],
                np.nonzero(neutral_words_remaining)[0],
                np.nonzero(danger_words_remaining)[0]
            )
        )

        # randint is inclusive on both sides
        guess = random.randint(0, len(possible_guesses)-1)

        return possible_guesses[guess]

    def generate_structured_guess(self, red_words_remaining, blue_words_remaining, neutral_words_remaining, danger_words_remaining, hint):
        possible_guesses = np.concatenate(
            (
                np.nonzero(red_words_remaining)[0],
                np.nonzero(blue_words_remaining)[0],
                np.nonzero(neutral_words_remaining)[0],
                np.nonzero(danger_words_remaining)[0]
            )
        )

        distances = []

        for i in range(len(possible_guesses)):
            hintWord = self.i2v[hint]
            guessWord = self.i2v[possible_guesses[i]]
            if hintWord in self.w2v and guessWord in self.w2v:
                distances.append(cosine(self.w2v[hintWord], self.w2v[guessWord]))
            else:
                distances.append(1)

        guessIndex = np.argmin(distances)

        return possible_guesses[guessIndex]