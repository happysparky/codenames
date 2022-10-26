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

''' move generate random hint to guesser and codemaster classes later?'''
def guesser_generate_random_guesses(game, count):
    unguessed_words = set(game.board()).difference(set(game.all_guesses))

    # return the maximum number of guesses possible for this team
    return random.sample(unguessed_words, count)
    

def codemaster_generate_random_hint(listOfWords, game):

    hint = random.choice(listOfWords)
    # ensure hint isn't in list of words
    while hint in game.board:
        hint = random.choice(listOfWords)

    words_remaining = len(game.red_words_remaining) if game.turn == 0 else len(game.blue_words_remaining)
    num_words = random.choice(range(1, words_remaining+1))

    return hint, num_words


def initialize_game(game, codemaster, listOfWords, batch_size):

    if type(codemaster) == HumanCodemaster:
        words_remaining = len(game.red_words_remaining)
        hint, count = get_humancodemaster_hint(codemaster, game)
    else:
        # the model starts off by picking generating a hint for a random number of words for the starting team
        state_init1 = game.get_state()  
        hint, count = codemaster_generate_random_hint(listOfWords, game)

    num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed = game.process_hint(hint, count)
    state_init2 = game.get_state()
    reward = codemaster.set_reward(num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed )
    codemaster.remember(state_init1, hint, reward, state_init2, game.crash)
    codemaster.replay_new(codemasterRed.memory, batch_size)  

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

        curCodemaster = params["codemasterRed"]
        curGuesser = params["guesserRed"]

        # perform first move
        initialize_game(game, curCodemaster, listOfWords, params["batch_size"])
        
        steps = 0       # steps since the last positive reward
        while (not game.crash) and (not game.end):
            if not params['train']:
                curCodemaster.epsilon = 0.01
                curGuesser.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                curCodemaster.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])
                curGuesser.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = game.get_state()

            # --- CODEMASTER ---
            # This should output 1 single word, w, and 1 integer, k
            # perform random actions based on agent.epsilon, or choose the action
            '''
            I used num_words before so I shouldn't be talking but we really need to come up with a more descriptive name
            than 'count'
            '''
            if random.uniform(0, 1) < curCodemaster.epsilon:
                hint, count = codemaster_generate_random_hint(listOfWords, game)
            else:
                # predict action based on the old state
                # TODO: should be able to add in a "bounding factor" for telling the model a min and max for the count output
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                    prediction = curCodemaster(state_old_tensor)
                    # TODO: generate word/number pair based on prediction
                    hint, count = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            count = game.process_hint(hint, count)
            state_new = game.get_state()

            # --- GUESSER ---
            numGuesses = 0
            guess = ""
            turnChanged = False
            '''
            two options: 
                1. generate all 5 words at once: pass in state_old_tensor and get a list of 5 words back
                2. generate 5 words one at a time: use a for loop to get the max likelihood word one at a time

            random chance to explore for the two above will look slightly different. I went with the first one 
            because the second one is slightly more difficult to implement I think
            '''
            # get maximum number of guesses possible amount of guesses back based on the hint
            if game.turn == 0:
                remaining = len(game.red_words_remaining)
            else:
                remaining = len(game.blue_words_remaining)

            if random.uniform(0, 1) < curGuesser.epsilon:
                guesses = guesser_generate_random_guesses(game)
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                    # TODO: generate guesses based on hint
                    # generate remaining number of guesses
                    guesses = curGuesser(state_old_tensor, hint, remaining)

            # try each guess 
            guessed_prev_correctly = True   
            for guess in guesses:
                # update the game state
                num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed = game.process_single_guess(guess)
                state_new = game.get_state()

                # set reward for the new state
                '''
                Add a huge reward for winning the game. 
                '''
                guesser_reward = curGuesser.set_reward(num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed)
                codemaster_reward = curCodemaster.set_reward(num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed)

                if params['train']:
                    # train short memory base on the new action and state
                    curCodemaster.train_short_memory(state_old, guess, reward, state_new, game.crash)
                    curGuesser.train_short_memory(state_old, guess, reward, state_new, game.crash)
                    # store the new data into a long term memory
                    curCodemaster.remember(state_old, guess, reward, state_new, game.crash)
                    curGuesser.remember(state_old, guess, reward, state_new, game.crash)

                # TODO: call model weight updates/loss/etc

                # preemptively this turn if failed to guess correctly or game ends
                if num_own_guessed == 0 or game.end:
                    break
            
            # if the game hasn't ended, change turns
            if game.end == False:
                if game.turn == 0:
                    game.turn = 1
                    curCodemaster = params["codemasterBlue"]
                    curGuesser = params["guesserBlue"]
                else:
                    game.turn = 0
                    curCodemaster = params["codemasterRed"]
                    curGuesser = params["guesserRed"]
            '''
            else:
                reward Codemaster and reward Guesser a ton for winning (can be done above in the regular reward function too) (pass in the game state?)
            '''

        # TODO: figure out what rewards to actually set -- this is for the codemaster only
        # TODO: need to add update weights etc.
        ''''
        look back in the snake code and figure what this does
        '''
        reward = curCodemaster.set_reward(game.state, game.crash)


        if params['train']:
            curCodemaster.replay_new(curCodemaster.memory, params['batch_size'])
            curGuesser.replay_new(curGuesser.memory, params['batch_size'])


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