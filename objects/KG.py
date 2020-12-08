import re
from objects.Entity import Entity
from objects.Relation import Relation
from objects.Attribute import Attribute
from objects.Literal import Literal


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

        self.relation_tuple_list = list()
        self.attribute_tuple_list = list()

        self.relation_set_func_ranked = list()
        self.relation_set_func_inv_ranked = list()
        self.attribute_set_func_ranked = list()
        self.attribute_set_func_inv_ranked = list()

        self.rel_or_attr_dict_by_tuple = dict()
        self.ent_or_lite_head_dict_by_tuple = dict()
        self.ent_or_lite_tail_dict_by_tuple = dict()

        self.__init()

    def __init(self):
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

        if self.ent_pre_func is None:
            self.ent_pre_func = default_pre_func
        if self.rel_pre_func is None:
            self.rel_pre_func = default_pre_func
        if self.attr_pre_func is None:
            self.attr_pre_func = default_pre_func
        if self.lite_pre_func is None:
            self.lite_pre_func = default_pre_func_for_literal

    @staticmethod
    def __dict_set_insert_helper(dictionary: dict, key, value):
        if dictionary.__contains__(key) is False:
            dictionary[key] = set()
        dictionary[key].add(value)

    def get_entity(self, name: str):
        if self.entity_dict_by_name.__contains__(name):
            return self.entity_dict_by_name.get(name)
        else:
            entity = Entity(idx=len(self.entity_set), name=name, preprocess_func=self.ent_pre_func, affiliation=self)
            self.entity_set.add(entity)
            self.entity_dict_by_name[entity.name] = entity
            self.entity_dict_by_value[entity.value] = entity
            return entity

    def get_relation(self, name: str):
        if self.relation_dict_by_name.__contains__(name):
            return self.relation_dict_by_name.get(name)
        else:
            relation = Relation(idx=len(self.relation_set), name=name, preprocess_func=self.rel_pre_func, affiliation=self)
            self.relation_set.add(relation)
            self.relation_dict_by_name[relation.name] = relation
            self.relation_dict_by_value[relation.value] = relation
            return relation

    def get_attribute(self, name: str):
        if self.attribute_dict_by_name.__contains__(name):
            return self.attribute_dict_by_name.get(name)
        else:
            attribute = Attribute(idx=len(self.attribute_set), name=name, preprocess_func=self.attr_pre_func, affiliation=self)
            self.attribute_set.add(attribute)
            self.attribute_dict_by_name[attribute.name] = attribute
            self.attribute_dict_by_value[attribute.value] = attribute
            return attribute

    def get_literal(self, name: str):
        if self.literal_dict_by_name.__contains__(name):
            return self.literal_dict_by_name.get(name)
        else:
            literal = Literal(name=name, preprocess_func=self.lite_pre_func, affiliation=self)
            self.literal_set.add(literal)
            self.literal_dict_by_name[literal.name] = literal
            self.literal_dict_by_value[literal.value] = literal
            return literal

    def insert_relation_tuple(self, head: str, relation: str, tail: str):
        ent_h, rel, ent_t = self.get_entity(head), self.get_relation(relation), self.get_entity(tail)
        ent_h.add_relation_as_head(relation=rel, tail=ent_t)
        rel.add_relation_tuple(head=ent_h, tail=ent_t)
        ent_t.add_relation_as_tail(relation=rel, head=ent_h)
        self.relation_tuple_list.append((ent_h, rel, ent_t))
        self.__dict_set_insert_helper(self.rel_or_attr_dict_by_tuple, (ent_h, ent_t), rel)
        self.__dict_set_insert_helper(self.ent_or_lite_head_dict_by_tuple, (ent_t, rel), ent_h)
        self.__dict_set_insert_helper(self.ent_or_lite_tail_dict_by_tuple, (ent_h, rel), ent_t)

    def insert_attribute_tuple(self, entity: str, attribute: str, literal: str):
        ent, attr, val = self.get_entity(entity), self.get_attribute(attribute), self.get_literal(literal)
        ent.add_attribute_tuple(attribute=attr, literal=val)
        attr.add_attribute_tuple(entity=ent, literal=val)
        val.add_attribute_tuple(entity=ent, attribute=attr)
        self.attribute_tuple_list.append((ent, attr, val))
        self.__dict_set_insert_helper(self.rel_or_attr_dict_by_tuple, (ent, val), attr)
        self.__dict_set_insert_helper(self.ent_or_lite_head_dict_by_tuple, (val, attr), ent)
        self.__dict_set_insert_helper(self.ent_or_lite_tail_dict_by_tuple, (ent, attr), val)

    def get_rel_or_attr_set_by_tuple(self, tup: tuple):
        return self.rel_or_attr_dict_by_tuple.get(tup, set())

    def get_ent_or_lite_head_set_by_tuple(self, tup: tuple):
        return self.ent_or_lite_head_dict_by_tuple.get(tup, set())

    def get_ent_or_lite_tail_set_by_tuple(self, tup: tuple):
        return self.ent_or_lite_tail_dict_by_tuple.get(tup, set())

    def calculate_functionality(self):
        for relation in self.relation_set:
            relation.calculate_functionality()
            self.relation_set_func_ranked.append(relation)
            self.relation_set_func_inv_ranked.append(relation)
        for attribute in self.attribute_set:
            attribute.calculate_functionality()
            self.attribute_set_func_ranked.append(attribute)
            self.attribute_set_func_inv_ranked.append(attribute)
        self.relation_set_func_ranked.sort(key=lambda x: x.functionality, reverse=True)
        self.relation_set_func_inv_ranked.sort(key=lambda x: x.functionality_inv, reverse=True)
        self.attribute_set_func_ranked.sort(key=lambda x: x.functionality, reverse=True)
        self.attribute_set_func_inv_ranked.sort(key=lambda x: x.functionality_inv, reverse=True)

    def print_kg_info(self):
        print("\nInformation of Knowledge Graph (" + str(self.name) + "):")
        print("- Relation Tuple Number: " + str(len(self.relation_tuple_list)))
        print("- Attribute Tuple Number: " + str(len(self.attribute_tuple_list)))
        print("- Entity Number: " + str(len(self.entity_set)))
        print("- Relation Number: " + str(len(self.relation_set)))
        print("- Attribute Number: " + str(len(self.attribute_set)))
        print("- Literal Number: " + str(len(self.literal_set)))
        print("- Functionality Statistics:")
        print("--- TOP-10 Relations (Func) ---")
        for i in range(min(10, len(self.relation_set_func_ranked))):
            relation = self.relation_set_func_ranked[i]
            print("Name: " + relation.name + "\tFunc: " + str(relation.functionality))

        print("--- TOP-10 Relations (Func-Inv) ---")
        for i in range(min(10, len(self.relation_set_func_inv_ranked))):
            relation = self.relation_set_func_inv_ranked[i]
            print("Name: " + relation.name + "\tFunc-Inv: " + str(relation.functionality_inv))

        print("--- TOP-10 Attributes (Func) ---")
        for i in range(min(10, len(self.attribute_set_func_ranked))):
            attribute = self.attribute_set_func_ranked[i]
            print("Name: " + attribute.name + "\tFunc: " + str(attribute.functionality))

        print("--- TOP-10 Attributes (Func-Inv) ---")
        for i in range(min(10, len(self.attribute_set_func_inv_ranked))):
            attribute = self.attribute_set_func_inv_ranked[i]
            print("Name: " + attribute.name + "\tFunc-Inv: " + str(attribute.functionality_inv))
