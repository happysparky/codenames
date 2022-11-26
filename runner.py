import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from HumanCodemaster import HumanCodemaster
from AgentCodemaster import AgentCodemaster
from HumanGuesser import HumanGuesser
from AgentGuesser import AgentGuesser
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

def print_board(game, i2v):
    print('--- BOARD ---')

    longest = 0
    for word_index in game.board:
        word = i2v[word_index]
        if len(word) > longest:
            longest = len(word)

    for idx in range(1, len(game.board)+1):
        if idx % 5 == 0: 
            end = "\n"
        else:
            end = " "

        current_word_index = game.board[idx-1]
        current_word = i2v[current_word_index]
        num_spaces = longest-len(current_word)

        if game.red_words_remaining[current_word_index] == 1:
            print(bcolors.RED + current_word + bcolors.ENDC, end=end)
        elif game.red_words_chosen[current_word_index] == 1: 
            print(bcolors.MAGENTA + current_word + bcolors.ENDC, end=end)
        elif game.blue_words_remaining[current_word_index] == 1:
            print(bcolors.BLUE + current_word + bcolors.ENDC, end=end)
        elif game.blue_words_chosen[current_word_index] == 1:
            print(bcolors.CYAN + current_word + bcolors.ENDC, end=end)
        elif game.neutral_words_remaining[current_word_index] == 1:
            print(bcolors.WHITE + current_word + bcolors.ENDC, end=end)
        elif game.neutral_words_chosen[ current_word_index] == 1: 
            print(bcolors.YELLOW + current_word + bcolors.ENDC, end=end)
        elif game.danger_words_remaining[current_word_index]: 
            print(bcolors.BLACK + current_word + bcolors.ENDC, end=end)

        if end == " ":
            for s in range(num_spaces):
                print(end=" ")

    # empty line for formatting purposes
    print()

def display(game, i2v):
    print_board(game, i2v)
    # print(game.blue_words_remaining + game.red_words_remaining + game.neutral_words_remaining + game.danger_word)
        
    print(bcolors.RED + "FOUND: " + str(indicesToWords(game.red_words_chosen, i2v)) + bcolors.ENDC)
    print(bcolors.RED + "LEFT: " + str(indicesToWords(game.red_words_remaining, i2v)) + bcolors.ENDC)
    print(bcolors.RED + "HINTS: " + str(indicesToWords(game.red_hints, i2v)) + bcolors.ENDC)
    
    print(bcolors.BLUE + "FOUND: " + str(indicesToWords(game.blue_words_chosen, i2v)) + bcolors.ENDC)
    print(bcolors.BLUE + "LEFT: " + str(indicesToWords(game.blue_words_remaining, i2v)) + bcolors.ENDC)
    print(bcolors.BLUE + "HINTS: " + str(indicesToWords(game.blue_hints, i2v)) + bcolors.ENDC)

    print(bcolors.WHITE + "FOUND: " + str(indicesToWords(game.neutral_words_chosen, i2v)) + bcolors.ENDC)
    print(bcolors.WHITE + "LEFT: " + str(indicesToWords(game.neutral_words_remaining, i2v)) + bcolors.ENDC)

    print(bcolors.BLACK + "DANGER: " + str(indicesToWords(game.danger_words_remaining, i2v)) + bcolors.ENDC)


    if game.turn == 0:
        print("\n" + bcolors.RED + "Red's Turn" + bcolors.ENDC)
    else:
        print("\n" + bcolors.BLUE + "Blue's Turn" + bcolors.ENDC)

# gets the hint and number of words the hint applies to. Ensures the hint and number of words the hint applies to is valid
def get_humancodemaster_hint(human_codemaster, game):
    gameWordBank = game.board
    words_remaining = game.red_words_remaining_count if game.turn == 0 else game.blue_words_remaining_count

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
        new_lines = [s.strip().lower() for s in f.readlines()]
        vocab_to_index = {w: i for i, w in enumerate(new_lines)}
        vocab_to_index["<UNK>"] = len(vocab_to_index)
        index_to_vocab = {i: w for i, w in enumerate(new_lines)}
        index_to_vocab[len(index_to_vocab)] = "<UNK>"
        return new_lines, vocab_to_index, index_to_vocab

def wordsToIndices(words, v2i):
    return [v2i[word] for word in words]

def indicesToWords(indices, i2v):
    return [i2v[i] for i in range(len(indices)) if indices[i] == 1]


def run(params, listOfWords, v2i, i2v):
    """
    Run the session, based on the parameters previously set.   
    """
       
    counter_games = 0
    score_plot = []
    counter_plot = []

    # play a certain number of games
    while counter_games < params['episodes']:
        # Initialize game state
        gameWordbank = random.sample(listOfWords, k=25)
        gameIndexbank = wordsToIndices(gameWordbank, v2i)
        game = Game(gameIndexbank, 8, len(v2i))

        curCodemaster = params["codemasterRed"]
        curGuesser = params["guesserRed"]
        
        '''
        look into what steps is used for
        '''
        steps = 0       # steps since the last positive reward
        while (not game.crash) and (not game.end) and steps < 200:
            # if logging, display board
            if params['display']:
                display(game, i2v)

            if not params['train']:
                curCodemaster.epsilon = 0.01
                curGuesser.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                # over the course of games, it will become smaller and smaller

                curCodemaster.epsilon = 0.3 - (counter_games * params['epsilon_decay_linear'])
                curGuesser.epsilon = 0.3 - (counter_games * params['epsilon_decay_linear'])

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

            if type(curCodemaster) != HumanCodemaster:
                randVal = random.uniform(0, 1)
                if randVal < curCodemaster.epsilon:
                    hint, count = game.generate_random_hint()
                else:
                    # predict action based on the old state
                    # TODO: should be able to add in a "bounding factor" for telling the model a min and max for the count output
                    with torch.no_grad():
                        codemaster_state_old_tensor = torch.from_numpy(codemaster_state_old).to(DEVICE)
                        codemaster_state_old_tensor = torch.flatten(codemaster_state_old_tensor)
            
                        hint_tensor, count_tensor = curCodemaster(codemaster_state_old_tensor)
                        # TODO: generate word/number pair based on prediction
                        hint = torch.argmax(hint_tensor).item()
                        count = torch.argmax(count_tensor).item()
                        if count > game.red_words_remaining_count and game.turn == 0:
                            count = game.red_words_remaining_count
                        elif count > game.blue_words_remaining_count and game.turn == 1:
                            count = game.blue_words_remaining_count
            else:
                hint, count = curCodemaster()
                if hint in v2i:
                    hint = v2i[hint]
                else:
                    hint = v2i["<UNK>"]

            # perform new move and get new state
            count = game.process_hint(hint, count)

            print("The hint given was '" + i2v[hint] + "' and it applies to " + str(count) + " words.")
            # get old state

            codemaster_state_new = game.get_codemaster_state()
            # print("GUESSER STATE", guesser_state_old.shape, guesser_state_old)

            # --- GUESSER ---


            # keep track of guesses made for codemaster and final guesser training
            accumulated_own_guessed = 0
            accumulated_opposing_guessed = 0
            accumulated_neutral_guessed = 0
            accumulated_danger_guessed = 0
            accumulated_previously_guessed = 0

            # can guess an extra word if all previous guesses for a hint are correct
            max_num_guesses = count + 1


            for idx in range(max_num_guesses):
                guesser_state_old = game.get_guesser_state()

                # generate a guess
                guess = ""

                # TODO: change implementation of how guesses are generated to take argmax instead of iterating through every 1
                if type(curGuesser) != HumanGuesser:
                    if random.uniform(0, 1) < curGuesser.epsilon:
                        guess = game.generate_random_guesses(1)
                    else:
                        # predict action based on the old state
                        with torch.no_grad():
                            guesser_state_old_tensor = torch.from_numpy(guesser_state_old).to(DEVICE)
                            guesser_state_old_tensor = torch.flatten(guesser_state_old_tensor)

                            # generate remaining number of guesses
                            guess = curGuesser(guesser_state_old_tensor)
                            guess = game.get_guesses_from_tensor(guess, 1)

                else:
                    guess = curGuesser()

                
                # update the game state
                num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed = game.process_single_guess(guess)
                accumulated_own_guessed += num_own_guessed
                accumulated_opposing_guessed += num_opposing_guessed
                accumulated_neutral_guessed += num_neutral_guessed
                accumulated_danger_guessed += num_danger_guessed
                accumulated_previously_guessed += num_previously_guessed

                guesser_state_new = game.get_guesser_state()

                # set reward for the new state
                guesser_reward = curGuesser.set_reward(num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed, game.end)

                # perform model weight updates/loss/etc
                if params['train']:
                    vocab_size = len(i2v)
                    guesser_state_old_tensor = torch.from_numpy(guesser_state_old)
                    guesser_state_new_tensor = torch.from_numpy(guesser_state_new)

                    # train short memory base on the new action and state
                    curGuesser.train_short_memory(guesser_state_old_tensor, guess, guesser_reward, guesser_state_new_tensor, game.crash)
                    # store the new data into a long term memory
                    curGuesser.remember(guesser_state_old_tensor, guess, guesser_reward, guesser_state_new_tensor, game.crash)

                # end this turn if failed to guess correctly or game ends
                if num_own_guessed == 0 or game.end:
                    print("Uh-oh! The word " + i2v[guess[0]] + " was incorrectly guessed!")
                    break
                else:
                    print("Good job! The word " + i2v[guess[0]] + " was correctly guessed!")
                if params['display']:
                    display(game, i2v)

            # guesser reward and weight updates
            ''' do we really have to reward the guesser again with accumulated stats? '''
            guesser_reward = curGuesser.set_reward(accumulated_own_guessed, accumulated_opposing_guessed, accumulated_neutral_guessed, accumulated_danger_guessed, accumulated_previously_guessed, game.end)
            codemaster_reward = curCodemaster.set_reward(accumulated_own_guessed, accumulated_opposing_guessed, accumulated_neutral_guessed, accumulated_danger_guessed, accumulated_previously_guessed, game.end)
            
            # if made good hints and guesses, steps is set to 0
            if codemaster_reward > 0 and guesser_reward > 0:
                steps = 0
                
            
            if params['train']:
                vocab_size = len(i2v)

                codemaster_state_old_tensor = torch.from_numpy(codemaster_state_old)
                codemaster_state_new_tensor = torch.from_numpy(codemaster_state_new)

                hint_tensor = np.zeros(vocab_size)
                hint_tensor[hint] = count
                hint_tensor = torch.from_numpy(hint_tensor)
                
                hint_tensorguesser_state_old_tensor = torch.from_numpy(guesser_state_old)
                guesser_state_new_tensor = torch.from_numpy(guesser_state_new)
                
                curCodemaster.train_short_memory(codemaster_state_old_tensor, hint_tensor, codemaster_reward, codemaster_state_new_tensor, game.crash)
                curCodemaster.remember(codemaster_state_old_tensor, hint_tensor, codemaster_reward, codemaster_state_new_tensor, game.crash)
                
                curGuesser.train_short_memory(hint_tensorguesser_state_old_tensor, guess, guesser_reward, guesser_state_new_tensor, game.crash)
                curGuesser.remember(hint_tensorguesser_state_old_tensor, guess, guesser_reward, guesser_state_new_tensor, game.crash)
            
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
            
            steps += 1
            
        if game.end:
            print("Game finished!")
            print("Winner: ", end="")

            if game.danger_words_remaining_count == 0:
                if game.turn == 0:
                    print("Blue Team")
                else:
                    print("Red Team")
            elif game.red_words_remaining_count == 0:
                    print("Red Team")
            else:
                print("Blue Team")
                
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
    
        if type(params["codemasterRed"]) == AgentCodemaster:
            torch.save(params["codemasterRed"].state_dict(), params["red_codemaster_weights"])
        if type(params["codemasterBlue"]) == AgentCodemaster:
            torch.save(params["codemasterBlue"].state_dict(), params["blue_codemaster_weights"])
        if type(params["guesserRed"]) == AgentGuesser:
            torch.save(params["guesserRed"].state_dict(), params["red_guesser_weights"])
        if type(params["guesserBlue"]) == AgentGuesser:
            torch.save(params["guesserBlue"].state_dict(), params["blue_guesser_weights"])


    return score_plot, counter_plot

def initialize_player(player, params, v2i, i2v, team):
    if player == HumanCodemaster:
        return HumanCodemaster()
    elif player == HumanGuesser:
        return HumanGuesser(v2i, i2v)
    elif player == AgentCodemaster:
        agent = AgentCodemaster(params, i2v, team)
        agent = agent.to(DEVICE).double()
        agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
        return agent
    elif player == AgentGuesser:
        agent = AgentGuesser(params, i2v, team)
        agent = agent.to(DEVICE).double()
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

    parser.add_argument("--red_codemaster_weights", help="where the weights of the red codemaster agent are stored", default="weights/RedCodemasterWeights")
    parser.add_argument("--blue_codemaster_weights", help="where the weights of the blue codemaster agent are stored", default="weights/BlueCodemasterWeights")
    parser.add_argument("--red_guesser_weights", help="where the weights of the red guesser agent are stored", default="weights/RedGuesserWeights")
    parser.add_argument("--blue_guesser_weights", help="where the weights of the blue guesser agent are stored", default="weights/BlueGuesserWeights")

    args = parser.parse_args()
    print("Args", args)

    # load codemaster classes
    codemasterRed = HumanCodemaster if args.codemasterRed else AgentCodemaster
    codemasterBlue = HumanCodemaster if args.codemasterBlue else AgentCodemaster

    # load guesser classes
    guesserRed = HumanGuesser if args.guesserRed else AgentGuesser
    guesserBlue = HumanGuesser if args.guesserBlue else AgentGuesser

    listOfWords, v2i, i2v = processWordbank('wordbank.txt')

    if params['train']:
        print("Training...")
        params['load_weights'] = False   # when training, the network is not pre-trained
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True

    params["red_codemaster_weights"] = args.red_codemaster_weights
    params["blue_codemaster_weights"] = args.blue_codemaster_weights
    params["red_guesser_weights"] = args.red_guesser_weights
    params['blue_guesser_weights'] = args.blue_guesser_weights

    params["codemasterRed"] = initialize_player(codemasterRed, params, v2i, i2v, 'red')
    params["codemasterBlue"] = initialize_player(codemasterBlue, params, v2i, i2v, 'blue')
    params["guesserRed"] = initialize_player(guesserRed, params, v2i, i2v, 'red')
    params["guesserBlue"] = initialize_player(guesserBlue, params, v2i, i2v, 'blue')

    score_plot, counter_plot = run(params, listOfWords, v2i, i2v)
    print(score_plot)
    print(counter_plot)