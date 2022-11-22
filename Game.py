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

    def __init__(self, wordList, per_team_count, vocab_size):

        self.vocab_size = vocab_size

        # sample |size| words
        self.red_words_remaining = np.zeros(vocab_size)
        for i in range(0,per_team_count+1):
            self.red_words_remaining[wordList[i]] = 1

        self.blue_words_remaining = np.zeros(vocab_size)
        for i in range(per_team_count+1,2*per_team_count+1):
            self.blue_words_remaining[wordList[i]] = 1

        self.neutral_words_remaining = np.zeros(vocab_size)
        for i in range(2*per_team_count+1,len(wordList)-1):
            self.neutral_words_remaining[wordList[i]] = 1
        self.danger_words_remaining = np.zeros(vocab_size)
        self.danger_words_remaining[wordList[len(wordList)-1]] = 1

        self.all_guesses = np.zeros(vocab_size)
        self.neutral_words_chosen = np.zeros(vocab_size)
        self.blue_words_chosen = np.zeros(vocab_size)
        self.blue_hints = np.zeros(vocab_size)
        self.red_words_chosen = np.zeros(vocab_size)
        self.red_hints = np.zeros(vocab_size)

        self.red_words_remaining_count = per_team_count+1
        self.blue_words_remaining_count = per_team_count
        self.danger_words_remaining_count = 1

        # shuffle wordlist to create board 
        random.shuffle(wordList) 
        self.board = wordList
        # 0 represents blue's turn, 1 is red's turn
        self.turn = 0
        # Score represents how many turns it takes to complete a game
        self.score = 0
        self.end = False

        ''' Set this to true somewhere'''
        self.crash = False

    '''
    Not sure what the codemaster's reward would be since it's dependent on how 
    many words are able to be guessed at the next step. 

    No direct comparison to the papers we were looking at before either since those didn't incorporate RL
    (but they still must have had some kind of measurement for their reward function? -- look into this)
    '''
    # Returns the max number of words that a hint might correspond to (based on which team's turn it is)
    def process_hint(self, hint, count):
        if self.turn == 0:
            self.red_hints[hint] = 1
            if count > self.red_words_remaining_count:
                count = self.red_words_remaining_count
        else:
            self.blue_hints[hint] = 1
            if count > self.blue_words_remaining_count:
                count = self.blue_words_remaining_count

        self.score += 1
        return count

    # makes guesses and returns metrics of how good the guess is 
    def process_single_guess(self, guess):
        num_previously_guessed = 0
        num_own_guessed = 0
        num_opposing_guessed = 0
        num_neutral_guessed = 0
        num_danger_guessed = 0

        print("guess:", guess)
        print(self.all_guesses)

        if self.all_guesses[guess] == 1:
            num_previously_guessed += 1
        else:
            self.all_guesses[guess] = 1

            if self.red_words_remaining[guess] == 1:
                self.red_words_remaining[guess] = 0
                self.red_words_remaining_count -= 1
                self.red_words_chosen[guess] = 1

                if self.turn == 0:
                    num_own_guessed += 1
                else:
                    num_opposing_guessed += 1

            elif self.blue_words_remaining[guess] == 1:
                self.blue_words_remaining[guess] = 0
                self.blue_words_remaining_count -= 1
                self.blue_words_chosen[guess] = 1

                if self.turn == 1:
                    num_own_guessed += 1
                else:
                    num_opposing_guessed += 1

            elif guess in self.neutral_words_remaining:
                self.neutral_words_remaining[guess] = 0
                self.neutral_words_chosen[guess] = 1

                num_neutral_guessed += 1

            elif self.danger_words_remaining[guess] == 1:
                self.end = True
                self.danger_words_remaining[guess] = 0
                self.danger_words_remaining_count -= 1
                num_danger_guessed += 1

            if self.blue_words_remaining_count == 0 or self.red_words_remaining_count == 0 or self.danger_words_remaining_count == 0:
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
            self.red_hints,
            self.red_words_chosen,
            self.red_words_remaining,
            self.blue_hints,
            self.blue_words_chosen,
            self.blue_words_remaining,
            self.neutral_words_chosen,
            self.neutral_words_remaining,
            self.danger_words_remaining,
            self.all_guesses
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
        remaining = np.zeros(self.vocab_size)
        remaining = self.red_words_remaining + self.blue_words_remaining + self.neutral_words_remaining + self.danger_words_remaining
        state = np.array([
            self.red_hints,
            self.red_words_chosen,
            self.blue_hints,
            self.blue_words_chosen,
            self.neutral_words_chosen,   
            self.all_guesses,
            remaining  
        ])

        return state

    # Create array of random guesses
    # Performs iterative guessing from all words on the board, checking to make sure that the word hasn't been guessed
    # Also makes sure that not all words have been guessed
    def generate_random_guesses(self, count):

        generated_guesses = []
        guessed = 0
        while len(generated_guesses) < count and guessed < len(self.board):
            guess = random.randint(0, self.vocab_size-1)
            if self.all_guesses[guess] == 0 and guess not in generated_guesses:
                generated_guesses.append(guess)
            guessed += 1

        # return the maximum number of guesses possible for this team
        return generated_guesses
        
    # Generates a random hint based on the vocabulary
    def generate_random_hint(self):

        hint = random.randint(0, self.vocab_size)
        # ensure hint isn't in list of words
        while hint in self.board:
            hint = random.randint(0, self.vocab_size)

        words_remaining = self.red_words_remaining_count if self.turn == 0 else self.blue_words_remaining_count
        num_words = random.choice(range(1, words_remaining+1))

        return hint, num_words
