
class Relation:
    def __init__(self):
        self.id: int
        self.name: str

        self.head_ent_set = set()
        self.tail_ent_set = set()

        self.functionality = 0
        self.functionality_inv = 0


