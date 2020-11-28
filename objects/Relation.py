
class Relation:
    def __init__(self, idx: int, name: str, affiliation=None):
        self.id: int = idx
        self.name: str = name
        self.affiliation = affiliation

        self.frequency = 0

        self.head_ent_set = set()
        self.tail_ent_set = set()

        self.functionality = 0.0
        self.functionality_inv = 0.0

    def add_relation_tuple(self, head, tail):
        self.head_ent_set.add(head)
        self.tail_ent_set.add(tail)
        self.frequency += 1

    def calculate_functionality(self):
        if self.frequency == 0:
            return
        self.functionality = len(self.head_ent_set) / self.frequency
        self.functionality_inv = len(self.tail_ent_set) / self.frequency

