import re


class Literal:
    def __init__(self, name: str, affiliation=None, pattern=r'"?([^"]*)"?(.*)'):
        self.name: str = name
        self.value = None
        self.affiliation = affiliation
        self.pattern = pattern

        self.involved_entity_set = set()
        self.involved_attr_set = set()
        self.__init()

    def __init(self):
        matchObj = re.match(self.pattern, self.name)
        self.value = matchObj.group(1)

    def add_attribute_tuple(self, entity, attribute):
        self.involved_entity_set.add(entity)
        self.involved_attr_set.add(attribute)
