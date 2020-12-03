class Entity:
    def __init__(self, idx: int, name: str, preprocess_func, affiliation=None):
        self._type = "ENTITY"
        self.id: int = idx
        self.name: str = name.strip()
        self.value = None
        self.preprocess_func = preprocess_func
        self.affiliation = affiliation

        self.involved_rel_dict = dict()
        self.involved_rel_inv_dict = dict()
        self.involved_attr_dict = dict()

        self.involved_as_tail_dict = dict()
        self.involved_as_head_dict = dict()

        self.neighbored_as_tail = set()

        self.__init()

    def __init(self):
        self.value = self.preprocess_func(self.name)

    def get_type(self):
        return self._type

    def add_relation_as_head(self, relation, tail):
        if self.involved_as_head_dict.__contains__(relation) is False:
            self.involved_rel_dict[relation] = set()
            self.involved_as_head_dict[relation] = set()

        self.involved_rel_dict[relation].add(tail)
        self.involved_as_head_dict[relation].add(tail)

    def add_relation_as_tail(self, relation, head):
        if self.involved_as_tail_dict.__contains__(relation) is False:
            self.involved_rel_inv_dict[relation] = set()
            self.involved_as_tail_dict[relation] = set()

        self.involved_rel_inv_dict[relation].add(head)
        self.involved_as_tail_dict[relation].add(head)
        self.neighbored_as_tail.add(head)

    def add_attribute_tuple(self, attribute, literal):
        if self.involved_as_head_dict.__contains__(attribute) is False:
            self.involved_as_head_dict[attribute] = set()
            self.involved_attr_dict[attribute] = set()

        self.involved_as_head_dict[attribute].add(literal)

