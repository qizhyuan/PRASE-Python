from objects.KG import KG
from objects.Entity import Entity
from objects.Relation import Relation
from config.PARISConfig import PARISConfig


class KGs:
    def __init__(self, kg1: KG, kg2: KG, paris_config=None):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = 0.1
        self.threshold = 0.8
        # self.PARIS_config = paris_config

        self.ent_align_candidate_dict = dict()
        self.rel_align_candidate_dict = dict()
        self.attr_align_candidate_dict = dict()
        self.lite_align_candidate_dict = dict()

        self.ent_align_refined_dict = dict()
        self.rel_align_refined_dict = dict()
        self.attr_align_refined_dict = dict()
        self.lite_align_refined_dict = dict()

        # self.refined_key_set = set()
        self.refined_tuple_dict = dict()

        self.__init()

    def __init(self):
        # if self.PARIS_config is None:
        #     self.PARIS_config = PARISConfig()

        for lite in self.kg_l.literal_set:
            if self.kg_r.literal_dict_by_value.__contains__(lite.value):
                self.lite_align_refined_dict[lite] = dict()
                self.lite_align_refined_dict[lite][self.kg_r.literal_dict_by_value[lite.value]] = 1.0

    def __run_per_iteration(self, init=False):


        return

    def __ent_or_lite_align(self, obj_l, obj_r, init=False):
        prob = 1.0
        involved_set_l = obj_l.involved_rel_set if obj_l.get_type() == "ENTITY" else obj_l.involved_attr_set
        involved_set_r = obj_r.involved_rel_set if obj_l.get_type() == "ENTITY" else obj_r.involved_attr_set
        for relation_l in involved_set_l:
            for relation_r in involved_set_r:
                if init is False and self.refined_tuple_dict.__contains__((relation_l, relation_r)) is False and \
                        self.refined_tuple_dict.__contains__((relation_r, relation_l)) is False:
                    continue
                for tail_l in obj_l.involved_rel_dict[relation_l]:
                    for tail_r in obj_r.involved_rel_dict[relation_r]:
                        if self.refined_tuple_dict.__contains__((tail_l, tail_r)) is False:
                            continue
                        equality = self.refined_tuple_dict.get((tail_l, tail_r), default=0)
                        p_lr = self.refined_tuple_dict.get((relation_l, relation_r),
                                                           default=0) if init is False else self.theta
                        p_rl = self.refined_tuple_dict.get((relation_r, relation_l),
                                                           default=0) if init is False else self.theta
                        prob *= (1.0 - p_lr * relation_r.functionality_inv * equality) * (1.0 - p_rl * relation_l.functionality_inv * equality)
        return 1.0 - prob

    def __rel_or_attr_align(self, obj_l: Relation, obj_r: Relation):
        numerator, denominator = 0.0, 0.0
        for (head_l, tail_l) in obj_l.tuple_set:
            num = 1.0
            for (head_r, tail_r) in obj_r.tuple_set:
                if self.refined_tuple_dict.__contains__((head_l, head_r)) is False or self.refined_tuple_dict.__contains__((tail_l, tail_r)) is False:
                    continue
                num *= 1.0 - self.refined_tuple_dict.get((head_l, head_r), default=0) * self.refined_tuple_dict.get((tail_l, tail_r), default=0)
            numerator += 1.0 - num

        for (head_l, tail_l) in obj_l.tuple_set:
            num = 1.0
            for head in (self.ent_align_refined_dict.get(head_l, default=set()) | self.lite_align_refined_dict.get(head_l, default=set())):
                for tail in (self.ent_align_refined_dict.get(tail_l, default=set()) | self.lite_align_refined_dict.get(tail_l, default=set())):
                    num *= 1.0 - self.refined_tuple_dict.get((head_l, head), default=0) * self.refined_tuple_dict.get(
                        (tail_l, tail), default=0)
                denominator += 1.0 - num



        






    def __get_align_prob(self, obj_l, obj_r):
        candidate = None
        if obj_l.get_type() == "ENTITY":
            if self.ent_align_refined_dict.__contains__(obj_l):
                candidate = self.ent_align_refined_dict[obj_l]
        if obj_l.get_type() == "RELATION":
            if self.rel_align_refined_dict.__contains__(obj_l):
                candidate = self.rel_align_refined_dict[obj_l]
        if obj_l.get_type() == "ATTRIBUTE":
            if self.attr_align_refined_dict.__contains__(obj_l):
                candidate = self.attr_align_refined_dict[obj_l]
        if obj_l.get_type() == "LITERAL":
            if self.lite_align_refined_dict.__contains__(obj_l):
                candidate = self.lite_align_refined_dict[obj_l]
        if candidate is None or obj_r not in candidate:
            return 0.0
        else:
            return candidate[obj_r]


    def print_alignment_result(self):
        print("\nAlignment Result:")
        print("--- Literal ---")
        for kg_l_lite in self.lite_align_refined_dict:
            candidate = self.lite_align_refined_dict[kg_l_lite]
            sorted(candidate.items(), key=lambda x: x[1], reverse=True)
            print(kg_l_lite.name + "\t" + list(candidate.keys())[0].name
                  + "\t" + str(list(candidate.values())[0]))



