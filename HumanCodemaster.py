from Codemaster import Codemaster

class HumanCodemaster(Codemaster):
    def __init__(self):
        super().__init__()
        return

    def forward(self):
        hint = input("Enter your hint: ")
        numWords = input("Enter the number of words this applies to: ")
        '''
        add checks to ensure input is fine
        '''
        numWords = int(numWords)
        return (hint, numWords)

    def postprocess(self, game):
        return

    def set_reward(self, num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed, game_ended):
        return

    def train_short_memory(self, state, action, reward, next_state, done):
        return
        