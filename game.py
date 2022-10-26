from tkinter.tix import INTEGER
import numpy as np
import matplotlib.pyplot as plt
from HumanCodemaster import HumanCodemaster
from Codemaster import Codemaster
from HumanGuesser import HumanGuesser
from Guesser import Guesser
from random import randint
import random
import torch.optim as optim
import argparse
import os
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
''' at some point go back and standardize upper camel case vs snake case'''

class Game:
    ''' 
    https://www.ultraboardgames.com/codenames/game-rules.php
    The starting team has 1 more word to guess. We arbitrarily chose red team to always starts first

    look into generalizing the above  -- does the starting team have an advantage like in chess? The 1 extra word to guess
    evens things out but still...

    The number of words guessed is also fixed to 9 and 8 - hardcode per_team_count? 
    '''

    def __init__(self, wordList, per_team_count):
        # sample |size| words
        self.red_words_remaining = wordList[:per_team_count+1] # choose per_team_count
        self.red_words_chosen = []
        self.red_hints = []

        self.blue_words_remaining = wordList[per_team_count+1:2*per_team_count+1] # choose per_team_count
        self.blue_words_chosen = []
        self.blue_hints = []


        self.neutral_words_remaining = wordList[2*per_team_count+1:len(wordList)-1]
        self.neutral_words_chosen = []
        self.danger_word = wordList[len(wordList)-1]

        self.all_guesses = []

        # shuffle wordlist to create board 
        random.shuffle(wordList) 
        self.board = wordList
        # 0 represents blue's turn, 1 is red's turn
        self.turn = 0
        self.end = False

        ''' Set this to true somewhere'''
        self.crash = False

    '''
    Not sure what the codemaster's reward would be since it's dependent on how 
    many words are able to be guessed at the next step. 

    No direct comparison to the papers we were looking at before either since those didn't incorporate RL
    (but they still must have had some kind of measurement for their reward function? -- look into this)
    '''
    def process_hint(self, hint, count):
        if self.turn == 0:
            self.red_hints.append(hint)
            if count > len(self.red_words_remaining):
                count = self.red_words_remaining
        else:
            self.blue_hints.append(hint) 
            if count > len(self.blue_words_remaining):
                count = self.blue_words_remaining
        return count

    # makes guesses and returns metrics of how good the guess is 
    def process_single_guess(self, guess):
        num_previously_guessed = 0
        num_own_guessed = 0
        num_opposing_guessed = 0
        num_neutral_guessed = 0
        num_danger_guessed = 0

        if guess in self.all_guesses:
            num_previously_guessed += 1
        else:
            self.all_guesses.append(guess)

            if guess in self.red_words_remaining:
                self.red_words_remaining.remove(guess)
                self.red_words_chosen.append(guess)

                if self.turn == 0:
                    num_own_guessed += 1
                else:
                    num_opposing_guessed += 1

            elif guess in self.blue_words_remaining:
                self.blue_words_remaining.remove(guess)
                self.blue_words_chosen.append(guess)

                if self.turn == 1:
                    num_own_guessed += 1
                else:
                    num_opposing_guessed += 1

            elif guess in self.neutral_words_remaining:
                self.neutral_words_remaining.remove(guess)
                self.neutral_words_chosen.append(guess)

                num_neutral_guessed += 1

            elif guess == self.danger_word:
                self.end = True
                num_danger_guessed += 1

            if len(self.blue_words_remaining) == 0 or len(self.red_words_remaining) == 0:
                self.end = True

        return num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed

    '''
    discuss how to represent the state
    - should we stratify based on turn vs including all info
        - stratifying based on turn means no adversial aspect, I think
    '''
    def get_codemaster_state(self):
        """
        Return the state.
        The state is a numpy array of 5 numpy arrays, representing:
            - red team's hints
            - red team's found words
            - blue team's guesses
            - blue team's found words
            - remaining words
        """
        state = [
            np.asarray(self.red_hints),
            np.asarray(self.red_words_chosen),
            np.asarray(self.red_words_remaining),
            np.asarray(self.blue_hints),
            np.asarray(self.blue_words_chosen),
            np.asarray(self.blue_words_remaining),
            np.asarray(self.neutral_words_chosen),
            np.asarray(self.neutral_words_remaining),
            np.asarray(self.danger_word),    
            np.asarray(self.all_guesses)        
        ]

        return np.asarray(state)

    '''
    discuss how to represent the state
    - should we stratify based on turn vs including all info
        - stratifying based on turn means no adversial aspect, I think
    '''
    def get_guesser_state(self):
        """
        Return the state.
        The state is a numpy array of 5 numpy arrays, representing:
            - red team's hints
            - red team's found words
            - blue team's guesses
            - blue team's found words
            - remaining words
        """
        remaining = sorted(self.blue_words_remaining + self.red_words_remaining + self.neutral_words_remaining + [self.danger_word])
        state = [
            np.asarray(self.red_hints),
            np.asarray(self.red_words_chosen),
            np.asarray(self.blue_hints),
            np.asarray(self.blue_words_chosen),
            np.asarray(self.neutral_words_chosen),   
            np.asarray(self.all_guesses),
            np.asarray(remaining)     
        ]

        return np.asarray(state)

    def generate_random_guesses(self, count):
        unguessed_words = set(self.board()).difference(set(self.all_guesses))

        # return the maximum number of guesses possible for this team
        return random.sample(unguessed_words, count)
        
    def generate_random_hint(self):

        hint = random.choice(self.board)
        # ensure hint isn't in list of words
        while hint in self.all_guesses:
            hint = random.choice(self.board)

        words_remaining = len(self.red_words_remaining) if self.turn == 0 else len(self.blue_words_remaining)
        num_words = random.choice(range(1, words_remaining+1))

        return hint, num_words
