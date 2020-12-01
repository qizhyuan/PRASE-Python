class Literal:
    def __init__(self, name: str, preprocess_func, affiliation=None):
        self._type = "LITERAL"
        self.name: str = name.strip()
        self.value = None
        self.preprocess_func = preprocess_func
        self.affiliation = affiliation

        self.involved_as_tail_dict = dict()
        self.involved_as_head_dict = dict()

        self.neighbored_as_tail = set()

        self.__init()

    def __init(self):
        self.value = self.preprocess_func(self.name)

    def get_type(self):
        return self._type

    def add_attribute_tuple(self, entity, attribute):
        if self.involved_as_tail_dict.__contains__(attribute) is False:
            self.involved_as_tail_dict[attribute] = set()
        self.involved_as_tail_dict[attribute].add(entity)
        self.neighbored_as_tail.add(entity)


