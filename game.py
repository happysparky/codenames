import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from HumanCodemaster import HumanCodemaster
from Codemaster import Codemaster
from HumanGuesser import HumanGuesser
from Guesser import Guesser
from random import randint
import random
import statistics
import torch.optim as optim
import torch
import datetime
import argparse
import os
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/100
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['episodes'] = 250          
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['train'] = False
    params["test"] = True
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params


class Game:
    """ Initialize PyGAME """
    ''' 
    https://www.ultraboardgames.com/codenames/game-rules.php
    The starting team has 1 more word to guess. We arbitrarily chose blue team to always starts first

    look into generalizing the above  -- does the starting team have an advantage like in chess? The 1 extra word to guess
    evens things out but still...

    The number of words guessed is also fixed to 9 and 8 - hardcode per_team_count? 
    '''

    def __init__(self, wordList, per_team_count):
        # sample |size| words
        self.blue_words_remaining = wordList[:per_team_count+1] # choose per_team_count
        self.blue_words_chosen = []
        self.blue_hints = []

        self.red_words_remaining = wordList[per_team_count+1:2*per_team_count+1] # choose per_team_count
        self.red_words_chosen = []
        self.red_hints = []
       
        self.neutral_words_remaining = wordList[2*per_team_count+1:len(wordList)-1]
        self.neutral_words_chosen = []
        self.danger_word = wordList[len(wordList)]

        # 0 represents blue's turn, 1 is red's turn
        self.turn = 0
    
    def do_hint(self, hint):
        if self.turn == 0:
            self.blue_hints.append(hint)
        else:
            self.red_hints.append(hint)
 
    def do_guess(self, guess):
        if guess in self.red_words_remaining:
            self.red_words_remaining.remove(guess)
            self.red_words_chosen.append(guess)
        elif guess in self.blue_words_remaining:
            self.blue_words_remaining.remove(guess)
            self.blue_words_chosen.append(guess)
        elif guess in self.neutral_words_remaining:
            self.neutral_words_remaining.remove(guess)
            self.neutral_words_chosen.append(guess)
        elif guess == self.danger_word:
            self.end = True
        self.turn = 1 if self.turn == 0 else 0

    def get_state(self):
        """
        Return the state.
        The state is a numpy array of 5 numpy arrays, representing:
            - own team's hints
            - own team's found words
            - opposing team's guesses
            - opposing team's found words
            - remaining words
        """
        state = [
            np.asarray(self.red_hints),
            np.asarray(self.red_words_chosen),
            np.asarray(self.blue_hints),
            np.asarray(self.blue_words_chosen),
            np.asarray(self.red_words_remaining + self.blue_words_remaining + self.neutral_words_chosen + self.danger_word),
        ]

        return np.asarray(state)

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def display(game):
    print('--- BOARD ---')
    print(game.blue_words_remaining + game.red_words_remaining + game.neutral_words_remaining + game.danger_word)
    print(bcolors.BLUE + "FOUND" + game.blue_words_chosen + bcolors.ENDC)
    print(bcolors.BLUE + "HINTS" + game.blue_hints + bcolors.ENDC)
    
    print(bcolors.RED + "FOUND" + game.red_words_chosen + bcolors.ENDC)
    print(bcolors.BLUE + "HINTS" + game.red_hints + bcolors.ENDC)

    if game.turn == 0:
        print("Red's Turn")
    else:
        print("Blue's Turn")


def initialize_game(player, game, food, agent, batch_size):
    state_init1 = game.get_state()  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = game.get_state()
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)  

def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = run(params)
    return score, mean, stdev

def processWordbank(filename):
    with open(filename, "r", encoding="utf-8") as f:
        new_lines = [s for s in f.readlines()]
        return new_lines

def run(params):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    listOfWords = processWordbank('wordbank.txt')

    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []

    # play a certain number of games
    while counter_games < params['episodes']:
        # Initialize game state
        gameWordbank = random.sample(listOfWords, k=30)
        game = Game(gameWordbank, 8)

        # set initial player
        team = "red"
        # if logging, display board
        if params['display']:
            display(game)

        # perform first move
        initialize_game(player1, game, agent, params['batch_size'])
        # TODO: need to assign player-per-turn and switch between turns
        # TODO: need to differentiate guesser vs codemaster agents
        
        steps = 0       # steps since the last positive reward
        while (not game.crash) and (not game.end):
            if not params['train']:
                agent.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = game.get_state()

            # --- CODEMASTER ---
            # This should output 1 single word, w, and 1 integer, k
            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = np.eye(3)[randint(0,2)]
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                    prediction = agent(state_old_tensor)
                    final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            game.do_hint(final_move, game)
            state_new = game.get_state()

            # set reward for the new state
            reward = agent.set_reward(player1, game.crash)

            # --- GUESSER ---
            numGuesses = 0
            guess = ""
            while numGuesses < k+1 and guess != game.danger_word:
                if random.uniform(0, 1) < agent.epsilon:
                    final_move = np.eye(3)[randint(0,2)]
                else:
                    # predict action based on the old state
                    with torch.no_grad():
                        state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                        prediction = agent(state_old_tensor)
                        final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

                # perform new move and get new state
                game.do_guess(final_move, game)
                state_new = game.get_state()

                # set reward for the new state
                reward = agent.set_reward(game.crash)
            if guess == game.danger_word:
                game.end = True

            # if team has won, set game.end = true
            if (len(game.blue_words_remaining)) == 0 or (len(game.red_words_remaining)) == 0:
                game.end = True
                
            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])

        counter_games += 1
        print(f'Game {counter_games}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(counter_games)

    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = define_parameters()

    parser.add_argument("--no_log", help="Supress logging", action='store_true', default=False)
    parser.add_argument("--no_print", help="Supress printing", action='store_true', default=False)
    parser.add_argument("--game_name", help="Name of game in log", default="default")

    args = parser.parse_args()
    print("Args", args)

    # load codemaster classes
    codemasterRed = HumanCodemaster if args.codemasterRed == "human" else Guesser
    codemasterBlue = HumanCodemaster if args.codemasterBlue == "human" else Codemaster

    # load guesser classes
    guesserRed = HumanGuesser if args.guesserRed == "human" else Guesser
    guesserBlue = HumanGuesser if args.guesserBlue == "human" else Guesser

    params['seed'] = randint()

    if params['train']:
        print("Training...")
        params['load_weights'] = False   # when training, the network is not pre-trained
        run(params)
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        run(params)