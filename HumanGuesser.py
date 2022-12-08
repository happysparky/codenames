from Guesser import Guesser
class HumanGuesser(Guesser):
    def __init__(self, v2i):
        super().__init__()
        self.v2i = v2i

    def forward(self):
        guess = input("Enter your guess: ")
        guess = guess.strip()
        guess = guess.lower()
        if guess in self.v2i:
            guess = self.v2i[guess]
        else:
            guess = self.v2i["<UNK>"]
        return guess


    

    