
class Attribute:
    def __init__(self):
        self.id: int
        self.name: str

        self.literal_set = set()

        self.functionality = 0
        self.functionality_inv = 0
