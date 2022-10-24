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


class Game:
    """ Initialize PyGAME """
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

        # shuffle wordlist to create board 
        random.shuffle(wordList) 
        self.board = wordList
        # 0 represents blue's turn, 1 is red's turn
        self.turn = 0
    
    '''
    Not sure what the codemaster's reward would be since it's dependent on how 
    many words are able to be guessed at the next step. 

    No direct comparison to the papers we were looking at before either since those didn't incorporate RL
    (but they still must have had some kind of measurement for their reward function? -- look into this)
    '''
    def do_hint(self, hint):
        if self.turn == 0:
            self.red_hints.append(hint)
        else:
            self.blue_hints.append(hint) 
 

    '''
    need to add checks (maybe in other functions, e.g. where we generate guesses or calculate the reward itself) for:
    - ensuring no repeated gusses within the same list of guesses. For example, if the hint is (ocean, 3), the AI shouldn't
    guess (fish, blue, fish)
    - keep track of history of guessed words (in the sense they've been chosen, not in the sense they've been correctly gussed) for 
    each team and apply a penalty if they've been guessed again  

    another note - do we have to worry about things in this fine detail? Or can we assume the model itself will learn that the optimal 
    move is to not repeat guesses? 
    - if we don't encode a representation for failed guesses we've done in the past, that's losing out on info. I think the model should be
    able to still learn the optimal move (ie no repetitions), but it might take more training time. 
    '''
    # makes guesses and returns metrics of how good the guess is 
    def do_guesses(self, guesses):
        num_own_guessed = 0
        num_opposing_guessed = 0
        num_neutral_guessed = 0
        num_danger_guessed = 0

        for guess in guesses:
            if guess in self.red_words_remaining:
                    self.red_words_remaining.remove(guess)
                    self.red_words_chosen.append(guess)

                    if self.turn == 0:
                        num_own_guessed += 1
                    else:
                        num_opposing_guessed += 1
                        # end turn if guessed opponent's word
                        break

            elif guess in self.blue_words_remaining:
                self.blue_words_remaining.remove(guess)
                self.blue_words_chosen.append(guess)

                if self.turn == 1:
                    num_own_guessed += 1
                else:
                    num_opposing_guessed += 1
                    # end turn if guessed opponent's word
                    break

            elif guess in self.neutral_words_remaining:
                self.neutral_words_remaining.remove(guess)
                self.neutral_words_chosen.append(guess)

                num_neutral_guessed += 1

            elif guess == self.danger_word:
                self.end = True
                num_danger_guessed += 1
                # end turn if gussed danger word
                break
            
        '''
        i think we might want to split this out
        '''
        # if no team has won or lost, change turns
        if (len(self.red_words_remaining) != 0) and (len(self.blue_words_remaining) != 0) and (not self.end):
            self.turn = 1 if self.turn == 0 else 0


        return num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed

    

    '''
    discuss how to represent the state
    - should we stratify based on turn vs including all info
        - stratifying based on turn means no adversial aspect, I think
    '''
    def get_state(self):
        """
        Return the state.
        The state is a numpy array of 5 numpy arrays, representing:
            - red team's hints
            - red team's found words
            - blue team's guesses
            - blue team's found words
            - remaining words
        """
        '''
        why are remaining words concatenated into one list?
        I separated them out because the agent needs to distinguish which words it needs to choose
        Kept the neutral words together because they don't matter too much
            - it's possible that there might be an effect on the agent 'knowing' a neutral word has already been chosen, so revist separating them out later
        '''
        state = [
            np.asarray(self.red_hints),
            np.asarray(self.red_words_chosen),
            np.asarray(self.red_words_remaining),
            np.asarray(self.blue_hints),
            np.asarray(self.blue_words_chosen),
            np.asarray(self.blue_words_remaining),
            np.asarray(self.neutral_words_chosen+self.neutral_words_remaining),
            np.asarray(self.danger_word),
            # why are these concatenated?
            # np.asarray(self.red_words_remaining + self.blue_words_remaining + self.neutral_words_chosen + [self.danger_word], dtype=object),
            
        ]

        return np.asarray(state)

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\u001b[37m'
    BLACK = '\u001b[30m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

    num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed = codemasterRed.do_hint(hint, num_words)
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
    Run the DQN algorithm, based on the parameters previously set.   
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
                game.do_guesses(final_move, game)
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