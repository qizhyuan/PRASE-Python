import re


class Literal:
    def __init__(self, name: str, preprocess_func, affiliation=None):
        self._type = "LITERAL"
        self.name: str = name.strip()
        self.value = None
        self.preprocess_func = preprocess_func
        self.affiliation = affiliation

        self.involved_entity_set = set()
        self.involved_attr_set = set()
        self.__init()

    def __init(self):
        self.value = self.preprocess_func(self.name)

    def get_type(self):
        return self._type

    def add_attribute_tuple(self, entity, attribute):
        self.involved_entity_set.add(entity)
        self.involved_attr_set.add(attribute)
