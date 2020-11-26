
class Literal:
    def __init__(self, name: str, affiliation=None):
        self.name: str = name
        self.affiliation = affiliation

        self.involved_entity_set = set()
        self.involved_attr_set = set()

    def add_attribute_tuple(self, entity, attribute):
        self.involved_entity_set.add(entity)
        self.involved_attr_set.add(attribute)
