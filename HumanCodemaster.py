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

        