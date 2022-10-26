class HumanCodemaster():
    def __init__(self):
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