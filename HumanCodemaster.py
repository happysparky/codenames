from Codemaster import Codemaster

class HumanCodemaster(Codemaster):
    def __init__(self):
        super().__init__()
        return

    def forward(self):

        hint = input("Enter your hint: ")

        count = "a"
        while not count.isnumeric():
            count = input("Enter the number of words this applies to: ")
        '''
        add checks to ensure input is fine
        '''
        count = int(count)
        hint = hint.lower()
        return (hint, count)

    def set_reward(self, num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_previously_guessed, game_ended):
        return 0

    def train_short_memory(self, state, action, reward, next_state, done):
        pass
        