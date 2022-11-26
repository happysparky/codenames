from Guesser import Guesser
class HumanGuesser(Guesser):
    def __init__(self, v2i, i2v):
        super().__init__()
        self.v2i = v2i
        self.i2v = i2v
        return

    def forward(self):
        guess = input("Enter your guess: ")
        guess = guess.strip()
        guess = guess.lower()
        if guess in self.v2i:
            guess = self.v2i[guess]
        else:
            guess = self.v2i["<UNK>"]
        return guess

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def replay_new(self, memory, batch_size):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass
    
    def set_reward(self, num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_prev_guessed, game_ended):
        return 0

    def get_state(self, game, player, food):
        pass

    

    