import re
import numpy as np

from objects.Entity import Entity
from objects.Relation import Relation


class KG:
    def __init__(self, name="KG", ent_pre_func=None, rel_pre_func=None, attr_pre_func=None,
                 lite_pre_func=None):
        self.name = name
        self.ent_pre_func = ent_pre_func
        self.rel_pre_func = rel_pre_func
        self.attr_pre_func = attr_pre_func
        self.lite_pre_func = lite_pre_func

        self.entity_set = set()
        self.relation_set = set()
        self.attribute_set = set()
        self.literal_set = set()

        self.entity_dict_by_name = dict()
        self.relation_dict_by_name = dict()
        self.attribute_dict_by_name = dict()
        self.literal_dict_by_name = dict()

        self.entity_dict_by_value = dict()
        self.relation_dict_by_value = dict()
        self.attribute_dict_by_value = dict()
        self.literal_dict_by_value = dict()

        self.entity_dict_by_id = dict()
        self.relation_dict_by_id = dict()

        self.relation_tuple_list = list()
        self.attribute_tuple_list = list()

        self.relation_set_func_ranked = list()
        self.relation_set_func_inv_ranked = list()
        self.attribute_set_func_ranked = list()
        self.attribute_set_func_inv_ranked = list()

        self.functionality_dict = dict()
        self.ent_id_list = list()
        self.fact_dict_by_head = dict()
        self.fact_dict_by_tail = dict()
        self.is_literal_list = list()

        self.ent_embeddings = None

        self.__init()

    def __init(self):
        if self.ent_pre_func is None:
            self.ent_pre_func = self.default_pre_func
        if self.rel_pre_func is None:
            self.rel_pre_func = self.default_pre_func
        if self.attr_pre_func is None:
            self.attr_pre_func = self.default_pre_func
        if self.lite_pre_func is None:
            self.lite_pre_func = self.default_pre_func_for_literal

    @staticmethod
    def default_pre_func(name: str):
        pattern = r'"?<?([^">]*)>?"?.*'
        matchObj = re.match(pattern=pattern, string=name)
        if matchObj is None:
            print("Match Error: " + name)
            return name
        value = matchObj.group(1).strip()
        if "/" in value:
            value = value.split(sep="/")[-1].strip()
        return value

    @staticmethod
    def default_pre_func_for_literal(name: str):
        value = name.split("^")[0].strip()
        start, end = 0, len(value) - 1
        if start < len(value) and value[start] == '<':
            start += 1
        if end > 0 and value[end] == '>':
            end -= 1
        if start < len(value) and value[start] == '"':
            start += 1
        if end > 0 and value[end] == '"':
            end -= 1
        if start > end:
            print("Match Error: " + name)
            return name
        value = value[start: end + 1].strip()
        return value

    @staticmethod
    def __dict_set_insert_helper(dictionary: dict, key, value):
        if dictionary.__contains__(key) is False:
            dictionary[key] = set()
        dictionary[key].add(value)

    def get_entity(self, name: str):
        if self.entity_dict_by_name.__contains__(name):
            return self.entity_dict_by_name.get(name)
        else:
            entity = Entity(idx=len(self.literal_set) + len(self.entity_set), name=name, preprocess_func=self.ent_pre_func, affiliation=self)
            self.entity_set.add(entity)
            self.entity_dict_by_name[entity.name] = entity
            self.entity_dict_by_value[entity.value] = entity
            self.entity_dict_by_id[entity.id] = entity
            self.ent_id_list.append(entity.id)
            self.is_literal_list.append(False)
            return entity

    def get_relation(self, name: str):
        if self.relation_dict_by_name.__contains__(name):
            return self.relation_dict_by_name.get(name)
        else:
            relation = Relation(idx=len(self.attribute_set) + len(self.relation_set), name=name, preprocess_func=self.rel_pre_func,
                                affiliation=self)
            self.relation_set.add(relation)
            self.relation_dict_by_name[relation.name] = relation
            self.relation_dict_by_value[relation.value] = relation
            self.relation_dict_by_id[relation.id] = relation
            return relation

    def get_attribute(self, name: str):
        if self.attribute_dict_by_name.__contains__(name):
            return self.attribute_dict_by_name.get(name)
        else:
            attribute = Relation(idx=len(self.attribute_set) + len(self.relation_set), name=name, preprocess_func=self.attr_pre_func,
                                 affiliation=self, is_attribute=True)
            self.attribute_set.add(attribute)
            self.attribute_dict_by_name[attribute.name] = attribute
            self.attribute_dict_by_value[attribute.value] = attribute
            self.relation_dict_by_id[attribute.id] = attribute
            return attribute

    def get_literal(self, name: str):
        if self.literal_dict_by_name.__contains__(name):
            return self.literal_dict_by_name.get(name)
        else:
            literal = Entity(idx=len(self.literal_set) + len(self.entity_set), name=name, preprocess_func=self.lite_pre_func,
                             affiliation=self, is_literal=True)
            self.literal_set.add(literal)
            self.literal_dict_by_name[literal.name] = literal
            self.literal_dict_by_value[literal.value] = literal
            self.entity_dict_by_id[literal.id] = literal
            self.is_literal_list.append(True)
            return literal

    def insert_relation_tuple(self, head: str, relation: str, tail: str):
        ent_h, rel, ent_t = self.get_entity(head), self.get_relation(relation), self.get_entity(tail)
        self.__insert_relation_tuple_one_way(ent_h, rel, ent_t)
        relation_inv = relation.strip() + str("-(INV)")
        rel_v = self.get_relation(relation_inv)
        self.__insert_relation_tuple_one_way(ent_t, rel_v, ent_h)

    def insert_attribute_tuple(self, entity: str, attribute: str, literal: str):
        ent, attr, val = self.get_entity(entity), self.get_attribute(attribute), self.get_literal(literal)
        self.__insert_attribute_tuple_one_way(ent, attr, val)
        attribute_inv = attribute.strip() + str("-(INV)")
        attr_v = self.get_attribute(attribute_inv)
        self.__insert_attribute_tuple_one_way(val, attr_v, ent)

    def __insert_relation_tuple_one_way(self, ent_h, rel, ent_t):
        ent_h.add_relation_as_head(relation=rel, tail=ent_t)
        rel.add_relation_tuple(head=ent_h, tail=ent_t)
        ent_t.add_relation_as_tail(relation=rel, head=ent_h)
        self.relation_tuple_list.append((ent_h, rel, ent_t))
        if not self.fact_dict_by_head.__contains__(ent_h.id):
            self.fact_dict_by_head[ent_h.id] = list()
        if not self.fact_dict_by_tail.__contains__(ent_t.id):
            self.fact_dict_by_tail[ent_t.id] = list()
        self.fact_dict_by_head[ent_h.id].append((rel.id, ent_t.id))
        self.fact_dict_by_tail[ent_t.id].append((rel.id, ent_h.id))

    def __insert_attribute_tuple_one_way(self, ent, attr, val):
        ent.add_relation_as_head(relation=attr, tail=val)
        attr.add_relation_tuple(head=ent, tail=val)
        val.add_relation_as_tail(relation=attr, head=ent)
        self.attribute_tuple_list.append((ent, attr, val))
        if not self.fact_dict_by_head.__contains__(ent.id):
            self.fact_dict_by_head[ent.id] = list()
        if not self.fact_dict_by_tail.__contains__(val.id):
            self.fact_dict_by_tail[val.id] = list()
        self.fact_dict_by_head[ent.id].append((attr.id, val.id))
        self.fact_dict_by_tail[val.id].append((attr.id, ent.id))

    def get_object_by_name(self, name: str):
        name = name.strip()
        if self.attribute_dict_by_name.__contains__(name):
            return self.attribute_dict_by_name[name]
        if self.relation_dict_by_name.__contains__(name):
            return self.relation_dict_by_name[name]
        if self.literal_dict_by_name.__contains__(name):
            return self.literal_dict_by_name[name]
        if self.entity_dict_by_name.__contains__(name):
            return self.entity_dict_by_name[name]

    def calculate_functionality(self):
        for relation in self.relation_set:
            relation.calculate_functionality()
            self.relation_set_func_ranked.append(relation)
            self.relation_set_func_inv_ranked.append(relation)
            self.functionality_dict[relation.id] = relation.functionality
        for attribute in self.attribute_set:
            attribute.calculate_functionality()
            self.attribute_set_func_ranked.append(attribute)
            self.attribute_set_func_inv_ranked.append(attribute)
            self.functionality_dict[attribute.id] = attribute.functionality
        self.relation_set_func_ranked.sort(key=lambda x: x.functionality, reverse=True)
        self.relation_set_func_inv_ranked.sort(key=lambda x: x.functionality_inv, reverse=True)
        self.attribute_set_func_ranked.sort(key=lambda x: x.functionality, reverse=True)
        self.attribute_set_func_inv_ranked.sort(key=lambda x: x.functionality_inv, reverse=True)

    def init_ent_embeddings(self):
        for ent in self.entity_set:
            idx, embedding = ent.id, ent.embedding
            if embedding is None:
                break
            if self.ent_embeddings is None:
                self.ent_embeddings = np.zeros((len(self.entity_set), len(embedding)))
            self.ent_embeddings[idx, :] = embedding

    def set_ent_embedding(self, idx, emb, func=None):
        if self.ent_embeddings is not None:
            if func is None:
                self.ent_embeddings[idx, :] = emb
            else:
                self.ent_embeddings[idx, :] = func(self.entity_dict_by_id[idx].embedding, emb)

    def print_kg_info(self):
        print("\nInformation of Knowledge Graph (" + str(self.name) + "):")
        print("- Relation Tuple Number: " + str(int(len(self.relation_tuple_list) / 2)))
        print("- Attribute Tuple Number: " + str(int(len(self.attribute_tuple_list) / 2)))
        print("- Entity Number: " + str(len(self.entity_set)))
        print("- Relation Number: " + str(int(len(self.relation_set) / 2)))
        print("- Attribute Number: " + str(int(len(self.attribute_set) / 2)))
        print("- Literal Number: " + str(len(self.literal_set)))
        print("- Functionality Statistics:")
        print("--- TOP-10 Relations (Func) ---")
        for i in range(min(10, len(self.relation_set_func_ranked))):
            relation = self.relation_set_func_ranked[i]
            print("Name: " + relation.name + "\tFunc: " + str(relation.functionality))
        print("......")
        print("--- TOP-10 Relations (Func-Inv) ---")
        for i in range(min(10, len(self.relation_set_func_inv_ranked))):
            relation = self.relation_set_func_inv_ranked[i]
            print("Name: " + relation.name + "\tFunc-Inv: " + str(relation.functionality_inv))
        print("......")
        print("--- TOP-10 Attributes (Func) ---")
        for i in range(min(10, len(self.attribute_set_func_ranked))):
            attribute = self.attribute_set_func_ranked[i]
            print("Name: " + attribute.name + "\tFunc: " + str(attribute.functionality))
        print("......")
        print("--- TOP-10 Attributes (Func-Inv) ---")
        for i in range(min(10, len(self.attribute_set_func_inv_ranked))):
            attribute = self.attribute_set_func_inv_ranked[i]
            print("Name: " + attribute.name + "\tFunc-Inv: " + str(attribute.functionality_inv))
        print("......")
