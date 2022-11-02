import os
import argparse
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

def display(game, i2v):
    print_board(game)
    # print(game.blue_words_remaining + game.red_words_remaining + game.neutral_words_remaining + game.danger_word)
        
    print(bcolors.RED + "FOUND: " + str(indicesToWords(game.red_words_chosen, i2v)) + bcolors.ENDC)
    print(bcolors.RED + "LEFT: " + str(indicesToWords(game.red_words_remaining, i2v)) + bcolors.ENDC)
    print(bcolors.RED + "HINTS: " + str(indicesToWords(game.red_hints, i2v)) + bcolors.ENDC)
    
    print(bcolors.BLUE + "FOUND: " + str(indicesToWords(game.blue_words_chosen, i2v)) + bcolors.ENDC)
    print(bcolors.BLUE + "LEFT: " + str(indicesToWords(game.blue_words_remaining, i2v)) + bcolors.ENDC)
    print(bcolors.BLUE + "HINTS: " + str(indicesToWords(game.blue_hints, i2v)) + bcolors.ENDC)

    print(bcolors.WHITE + "FOUND: " + str(indicesToWords(game.neutral_words_chosen, i2v)) + bcolors.ENDC)
    print(bcolors.WHITE + "LEFT: " + str(indicesToWords(game.neutral_words_remaining, i2v)) + bcolors.ENDC)

    print(bcolors.BLACK + "DANGER: " + str(indicesToWords([game.danger_word], i2v)) + bcolors.ENDC)


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
    while (hint in gameWordBank) or (num_words < 1) or (num_words > words_remaining) or (not isinstance(num_words, int)):
        if hint in gameWordBank:
            print(hint + " is on the board, please come up with a different hint. ")

        else:
            print("The number of words this hint applies to is invalid. Please ensure that it is an integer, greater than 0, and \
            less than or equal to " + str(words_remaining) + ", the number of words left to guess for your team. ")
        hint, num_words = human_codemaster.forward()

    return hint, num_words

def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = run(params)
    return score, mean, stdev

def processWordbank(filename):
    with open(filename, "r", encoding="utf-8") as f:
        new_lines = [s.strip() for s in f.readlines()]
        vocab_to_index = {w: i for i, w in enumerate(new_lines)}
        index_to_vocab = {i: w for i, w in enumerate(new_lines)}
        return new_lines, vocab_to_index, index_to_vocab

def wordsToIndices(words, v2i):
    return [v2i[word] for word in words]

def indicesToWords(indices, i2v):
    return [i2v[index] for index in indices]

def run(params):
    """
    Run the session, based on the parameters previously set.   
    """

    listOfWords, v2i, i2v = processWordbank('wordbank.txt')
       
    counter_games = 0
    score_plot = []
    counter_plot = []

    # play a certain number of games
    while counter_games < params['episodes']:
        # Initialize game state
        gameWordbank = random.sample(listOfWords, k=25)
        gameIndexbank = wordsToIndices(gameWordbank, v2i)
        game = Game(gameIndexbank, 8)

        # if logging, display board
        if params['display']:
            display(game, i2v)

        curCodemaster = params["codemasterRed"]
        curGuesser = params["guesserRed"]
        
        '''
        look into what steps is used for
        '''
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
            codemaster_state_old = game.get_codemaster_state()

            # --- CODEMASTER ---
            # This should output 1 single word, w, and 1 integer, k
            # perform random actions based on agent.epsilon, or choose the action
            # TODO: need to print out board if logging so that human codemaster can see it
            '''
            I used num_words before so I shouldn't be talking but we really need to come up with a more descriptive name
            than 'count'
            '''
            if random.uniform(0, 1) < curCodemaster.epsilon:
                hint, count = game.generate_random_hint()
            else:
                # predict action based on the old state
                # TODO: should be able to add in a "bounding factor" for telling the model a min and max for the count output
                with torch.no_grad():
                    codemaster_state_old_tenser = torch.tensor(codemaster_state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                    prediction = curCodemaster(codemaster_state_old_tenser)
                    # TODO: generate word/number pair based on prediction
                    hint, count = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            count = game.process_hint(hint, count)
            # get old state

            codemaster_state_new = game.get_codemaster_state()
            guesser_state_old = game.get_guesser_state()

            # --- GUESSER ---
            guess = ""
            # get maximum number of guesses possible amount of guesses back based on the hint
            if game.turn == 0:
                remaining = len(game.red_words_remaining)
            else:
                remaining = len(game.blue_words_remaining)

            if random.uniform(0, 1) < curGuesser.epsilon:
                guesses = game.generate_random_guesses(remaining)
            else:
                # predict action based on the old state
                with torch.no_grad():
                    guesser_state_old_tensor = torch.tensor(guesser_state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                    # TODO: generate guesses based on hint
                    # generate remaining number of guesses
                    guesses = curGuesser(guesser_state_old_tensor, hint, remaining)

            # try each guess 
            accumulated_own_guessed = 0
            accumulated_opposing_guessed = 0
            accumulated_neutral_guessed = 0
            accumulated_danger_guessed = 0
            accumulated_previously_guessed = 0
            for guess in guesses:
                # update the game state
                num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed = game.process_single_guess(guess)
                accumulated_own_guessed += num_own_guessed
                accumulated_opposing_guessed += num_opposing_guessed
                accumulated_neutral_guessed += num_neutral_guessed
                accumulated_danger_guessed += num_danger_guessed
                accumulated_previously_guessed += num_previously_guessed

                guesser_state_new = game.get_guesser_state()

                # set reward for the new state
                codemaster_reward = curCodemaster.set_reward(num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed, game.end)
                guesser_reward = curGuesser.set_reward(num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed, game.end)

                # perform model weight updates/loss/etc
                if params['train']:
                    # train short memory base on the new action and state
                    curGuesser.train_short_memory(guesser_state_old, guess, guesser_reward, guesser_state_new, game.crash)
                    # store the new data into a long term memory
                    curGuesser.remember(guesser_state_old, guess, guesser_reward, guesser_state_new, game.crash)

                # end this turn if failed to guess correctly or game ends
                if num_own_guessed == 0 or game.end:
                    break

            # guesser reward and weight updates
            guesser_reward = curGuesser.set_reward(accumulated_own_guessed, accumulated_opposing_guessed, accumulated_neutral_guessed, accumulated_danger_guessed, accumulated_previously_guessed, game.end)
            codemaster_reward = curCodemaster.set_reward(accumulated_own_guessed, accumulated_opposing_guessed, accumulated_neutral_guessed, accumulated_danger_guessed, accumulated_previously_guessed, game.end)
            if params['train']:
                curCodemaster.train_short_memory(codemaster_state_old, hint, count, codemaster_reward, codemaster_state_new, game.crash)
                curCodemaster.remember(codemaster_state_old, hint, count, codemaster_reward, codemaster_state_new, game.crash)
                curGuesser.train_short_memory(guesser_state_old, guess, guesser_reward, guesser_state_new, game.crash)
                curGuesser.remember(guesser_state_old, guess, guesser_reward, guesser_state_new, game.crash)
            
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
        different batch size for codemaster and guesser?
        '''
        if params['train']:
            curCodemaster.replay_new(curCodemaster.memory, params['batch_size'])
            curGuesser.replay_new(curGuesser.memory, params['batch_size'])

        # TODO: should be calculating some sort of accuracy
        counter_games += 1
        print(f'Game {counter_games}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(counter_games)

    if params['train']:
        codemaster_model_weights = curCodemaster.state_dict()
        torch.save(codemaster_model_weights, params["weights_path"])
        
        guesser_model_weights = curGuesser.state_dict()
        torch.save(guesser_model_weights, params["weights_path"])

    return score_plot, counter_plot

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

    score_plot, counter_plot = run(params)
    print(score_plot)
    print(counter_plot)