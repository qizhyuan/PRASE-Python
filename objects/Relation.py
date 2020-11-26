
class Relation:
    def __init__(self, idx: int, name: str, affiliation=None):
        self.id: int = idx
        self.name: str = name
        self.affiliation = affiliation

        self.head_ent_set = set()
        self.tail_ent_set = set()

        self.functionality = 0
        self.functionality_inv = 0

    def add_relation_tuple(self, head, tail):
        self.head_ent_set.add(head)
        self.tail_ent_set.add(tail)

