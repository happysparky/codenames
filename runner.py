import os
import argparse
from tkinter.tix import INTEGER
import numpy as np
import matplotlib.pyplot as plt
from HumanCodemaster import HumanCodemaster
from Codemaster import Codemaster
from HumanGuesser import HumanGuesser
from Guesser import Guesser
from random import randint
from Game import Game
from bcolors import bcolors
import random
import statistics
import torch.optim as optim
import torch
import datetime
import argparse
import os
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
''' at some point go back and standardize upper camel case vs snake case'''

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
    params['episodes'] = 1
    # params['episodes'] = 250          
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    # Setting train to true and test to false for now
    params['train'] = True
    params["test"] = False
    params['plot_score'] = True
    params['display'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params

def print_board(game):
    print('--- BOARD ---')

    longest = 0
    for word in game.board:
        if len(word) > longest:
            longest = len(word)

    for idx in range(1, len(game.board)+1):
        if idx % 5 == 0: 
            end = "\n"
        else:
            end = " "

        current_word = game.board[idx-1]
        num_spaces = longest-len(current_word)

        if current_word in game.red_words_remaining or current_word in game.red_words_chosen: 
            print(bcolors.RED + current_word + bcolors.ENDC, end=end)
        elif current_word in game.blue_words_remaining or current_word in game.blue_words_chosen:
            print(bcolors.BLUE + current_word + bcolors.ENDC, end=end)
        elif current_word in game.neutral_words_remaining or current_word in game.neutral_words_chosen: 
            print(bcolors.WHITE + current_word + bcolors.ENDC, end=end)
        elif current_word in game.danger_word: 
            print(bcolors.BLACK + current_word + bcolors.ENDC, end=end)

        if end == " ":
            for s in range(num_spaces):
                print(end=" ")

    # empty line for formatting purposes
    print()

def display(game):
    print_board(game)
    # print(game.blue_words_remaining + game.red_words_remaining + game.neutral_words_remaining + game.danger_word)
        
    print(bcolors.RED + "FOUND: " + str(game.red_words_chosen) + bcolors.ENDC)
    print(bcolors.RED + "LEFT: " + str(game.red_words_remaining) + bcolors.ENDC)
    print(bcolors.RED + "HINTS: " + str(game.red_hints) + bcolors.ENDC)
    
    print(bcolors.BLUE + "FOUND: " + str(game.blue_words_chosen) + bcolors.ENDC)
    print(bcolors.BLUE + "LEFT: " + str(game.blue_words_remaining) + bcolors.ENDC)
    print(bcolors.BLUE + "HINTS: " + str(game.blue_hints) + bcolors.ENDC)

    print(bcolors.WHITE + "FOUND: " + str(game.neutral_words_chosen) + bcolors.ENDC)
    print(bcolors.WHITE + "LEFT: " + str(game.neutral_words_remaining) + bcolors.ENDC)

    print(bcolors.BLACK + "DANGER: " + str(game.danger_word) + bcolors.ENDC)


    if game.turn == 0:
        print(bcolors.RED + "Red's Turn" + bcolors.ENDC)
    else:
        print(bcolors.BLUE + "Blue's Turn" + bcolors.ENDC)

# gets the hint and number of words the hint applies to. Ensures the hint and number of words the hint applies to is valid
def get_humancodemaster_hint(human_codemaster, game):
    gameWordBank = game.board
    words_remaining = len(game.red_words_remaining) if game.turn == 0 else len(game.blue_words_remaining)

    hint, num_words = human_codemaster.forward()

    # the hint can't be a word on the board
    # the number of words the hint applies to has to be > 0,  <= number of words remaining for the team, and an integer
    while (hint in gameWordBank) or (num_words < 1) or (num_words > len(words_remaining)) or (not isinstance(num_words, int)):
        if hint in gameWordBank:
            print(hint + " is on the board, please come up with a different hint. ")

        else:
            print("The number of words this hint applies to is invalid. Please ensure that it is an integer, greater than 0, and \
            less than or equal to " + str(words_remaining) + ", the number of words left to guess for your team. ")
        hint, num_words = human_codemaster.forward()

    return hint, num_words


def codemaster_generate_random_hint(listOfWords, game):

    hint = random.choice(listOfWords)
    # ensure hint isn't in list of words
    while hint in game.board:
        hint = random.choice(listOfWords)

    words_remaining = len(game.red_words_remaining) if game.turn == 0 else len(game.blue_words_remaining)
    num_words = random.choice(range(1, words_remaining+1))

    return hint, num_words


def initialize_game(game, codemasterRed, listOfWords, batch_size):

    if type(codemasterRed) == HumanCodemaster:
        words_remaining = len(game.red_words_remaining)
        hint, num_words = get_humancodemaster_hint(codemasterRed, game)
    else:
        # the model starts off by picking generating a hint for a random number of words for the starting team
        state_init1 = game.get_state()  
        hint, num_words = codemaster_generate_random_hint(listOfWords, game)

    num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed = codemasterRed.process_hint(hint, num_words)
    state_init2 = game.get_state()
    reward = codemasterRed.set_reward(num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed )
    codemasterRed.remember(state_init1, hint, reward, state_init2, game.crash)
    codemasterRed.replay_new(codemasterRed.memory, batch_size)  

def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = run(params)
    return score, mean, stdev

def processWordbank(filename):
    with open(filename, "r", encoding="utf-8") as f:
        new_lines = [s.strip() for s in f.readlines()]
        return new_lines

def run(params):
    """
    Run the session, based on the parameters previously set.   
    """

    listOfWords = processWordbank('wordbank.txt')
       
    counter_games = 0
    score_plot = []
    counter_plot = []

    # play a certain number of games
    while counter_games < params['episodes']:
        # Initialize game state
        gameWordbank = random.sample(listOfWords, k=25)
        game = Game(gameWordbank, 8)

        # if logging, display board
        if params['display']:
            display(game)

        # perform first move
        initialize_game(game, params["codemasterRed"], listOfWords, params["batch_size"])
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
                # TODO: should be able to add in a "bounding factor" for telling the model a min and max for the count output
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                    prediction = agent(state_old_tensor)
                    # TODO: generate word/number pair based on prediction
                    hint, count = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            count = game.process_hint(hint, count)
            state_new = game.get_state()

            # --- GUESSER ---
            numGuesses = 0
            guess = ""
            turnChanged = False
            while numGuesses < count+1 and guess != game.danger_word and game.end == False:
                if random.uniform(0, 1) < agent.epsilon:
                    final_move = np.eye(3)[randint(0,2)]
                else:
                    # predict action based on the old state
                    with torch.no_grad():
                        state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                        # TODO: generate single-word guess based on prediction
                        prediction = agent(state_old_tensor)
                        final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

                # perform new move and get new state
                num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed = game.process_single_guess(final_move)
                state_new = game.get_state()

                # set reward for the new state
                reward = agent.set_reward(num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed)

                # TODO: call model weight updates/loss/etc

                # check if changing turns
                if num_own_guessed == 0 or (len(game.blue_words_remaining)) == 0 or (len(game.red_words_remaining)) == 0:
                    break

            # change turn
            self.turn = 1 if self.turn == 0 else 0
            # TODO: note need to change agents being used

            # if team has won, set game.end = true
            if (len(game.blue_words_remaining)) == 0 or (len(game.red_words_remaining)) == 0:
                game.end = True
                
            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)


        # TODO: figure out what rewards to actually set -- this is for the codemaster only
        # TODO: need to add update weights etc.
        reward = agent.set_reward(player1, game.crash)


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

def initialize_player(player, params):
    if player == HumanCodemaster:
        return HumanCodemaster()
    elif player == HumanGuesser:
        return HumanGuesser()
    elif player == Codemaster:
        agent = Codemaster(params)
        agent = agent.to(DEVICE)
        agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
        return agent
    else:
        agent = Guesser(params)
        agent = agent.to(DEVICE)
        agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
        return agent 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = define_parameters()
    
    parser.add_argument("--codemasterRed", help="include this flag if you want this role to be played by a human", action="store_true")   
    parser.add_argument("--codemasterBlue", help="include this flag if you want this role to be played by a human", action="store_true") 
    parser.add_argument("--guesserRed", help="include this flag if you want this role to be played by a human", action="store_true")   
    parser.add_argument("--guesserBlue", help="include this flag if you want this role to be played by a human", action="store_true")

    parser.add_argument("--no_log", help="Supress logging", action='store_true', default=False)
    parser.add_argument("--no_print", help="Supress printing", action='store_true', default=False)
    parser.add_argument("--game_name", help="Name of game in log", default="default")
    
    args = parser.parse_args()
    print("Args", args)

    # load codemaster classes
    codemasterRed = HumanCodemaster if args.codemasterRed else Guesser
    codemasterBlue = HumanCodemaster if args.codemasterBlue else Codemaster

    # load guesser classes
    guesserRed = HumanGuesser if args.guesserRed else Guesser
    guesserBlue = HumanGuesser if args.guesserBlue else Guesser

    '''
    Not sure what this does so commenting it out for now
    Also randint() requires positional arguments
    '''
    # params['seed'] = randint()

    if params['train']:
        print("Training...")
        params['load_weights'] = False   # when training, the network is not pre-trained
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True

    params["codemasterRed"] = initialize_player(codemasterRed, params)
    params["codemasterBlue"] = initialize_player(codemasterBlue, params)
    params["guesserRed"] = initialize_player(guesserRed, params)
    params["guesserBlue"] = initialize_player(guesserBlue, params)

    run(params)