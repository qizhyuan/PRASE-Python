
class Attribute:
    def __init__(self, idx: int, name: str, preprocess_func, affiliation=None):
        self._type = "ATTRIBUTE"
        self.id: int = idx
        self.name: str = name.strip()
        self.value = None
        self.preprocess_func = preprocess_func
        self.affiliation = affiliation

        self.frequency = 0

        self.head_ent_set = set()
        self.tail_ent_set = set()
        self.tuple_set = set()

        self.functionality = 0.0
        self.functionality_inv = 0.0
        self.__init()

    def __init(self):
        self.value = self.preprocess_func(self.name)

    def get_type(self):
        return self._type

    def add_attribute_tuple(self, head, tail):
        self.head_ent_set.add(head)
        self.tail_ent_set.add(tail)
        self.tuple_set.add((head, tail))
        self.frequency += 1

    def calculate_functionality(self):
        if self.frequency == 0:
            return
        self.functionality = len(self.head_ent_set) / self.frequency
        self.functionality_inv = len(self.tail_ent_set) / self.frequency
