from objects import KnowledgeGraph


class Entity:
    def __init__(self):
        self.id: int
        self.name: str
        self.affiliation: KnowledgeGraph
        self.counterpart: Entity

        self.involved_rel_set = set()
        self.involved_rel_inv_set = set()
        self.involved_attr_set = set()

        self.involved_rel_dict = dict()
        self.involved_rel_inv_dict = dict()
        self.involved_attr_dict = dict()
