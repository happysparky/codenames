class HumanGuesser():
    def __init__(self):
        super().__init__()
        return

    def forward(self):
        guess = input("Enter your guess: ")
        guess = guess.strip()
        guess = guess.lower()
        return guess

    def train_short_memory(self, state, action, reward, next_state, done):
        return

    def replay_new(self, memory, batch_size):
        return

    def remember(self, state, action, reward, next_state, done):
        return
    
    def set_reward(self, num_own_guessed, num_opposing_guessed, num_neutral_guessed, num_danger_guessed, num_prev_guessed, game_ended):
        return

    def get_state(self, game, player, food):
        return

    

    