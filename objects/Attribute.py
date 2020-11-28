
class Attribute:
    def __init__(self, idx: int, name: str, affiliation=None):
        self.id: int = idx
        self.name: str = name
        self.affiliation = affiliation

        self.frequency = 0

        self.entity_set = set()
        self.literal_set = set()

        self.functionality = 0.0
        self.functionality_inv = 0.0

    def add_attribute_tuple(self, entity, literal):
        self.entity_set.add(entity)
        self.literal_set.add(literal)
        self.frequency += 1

    def calculate_functionality(self):
        if self.frequency == 0:
            return
        self.functionality = len(self.entity_set) / self.frequency
        self.functionality_inv = len(self.literal_set) / self.frequency
