import re


class Entity:
    def __init__(self, idx: int, name: str, preprocess_func, affiliation=None):
        self._type = "ENTITY"
        self.id: int = idx
        self.name: str = name.strip()
        self.value = None
        self.preprocess_func = preprocess_func
        self.affiliation = affiliation
        # self.counterpart = counterpart

        self.involved_rel_set = set()
        self.involved_rel_inv_set = set()
        self.involved_attr_set = set()

        self.involved_rel_dict = dict()
        self.involved_rel_inv_dict = dict()
        self.involved_attr_dict = dict()

        self.neighbor_set = set()
        self.neighbor_set_inv = set()

        self.__init()

    def __init(self):
        self.value = self.preprocess_func(self.name)

    def get_type(self):
        return self._type

    def add_relation_as_head(self, relation, tail):
        if self.involved_rel_set.__contains__(relation) is False:
            self.involved_rel_set.add(relation)
            self.involved_rel_dict[relation] = set()

        self.involved_rel_dict[relation].add(tail)
        self.neighbor_set.add(tail)

    def add_relation_as_tail(self, relation, head):
        if self.involved_rel_inv_set.__contains__(relation) is False:
            self.involved_rel_inv_set.add(relation)
            self.involved_rel_inv_dict[relation] = set()

        self.involved_rel_inv_dict[relation].add(head)
        self.neighbor_set_inv.add(head)

    def add_attribute_tuple(self, attribute, literal):
        if self.involved_attr_set.__contains__(attribute) is False:
            self.involved_attr_set.add(attribute)
            self.involved_attr_dict[attribute] = set()

        self.involved_attr_dict[attribute].add(literal)

