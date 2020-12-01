from objects.KG import KG


class KGs:
    def __init__(self, kg1: KG, kg2: KG, paris_config=None):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = 0.1
        self.threshold = 0.1
        self.ent_lite_candidate_num = 5
        self.rel_attr_candidate_num = 3
        self.iteration = 1
        # self.PARIS_config = paris_config
        self.ent_lite_align_candidate_dict = dict()
        self.rel_attr_align_candidate_dict = dict()

        self.ent_lite_align_refined_dict = dict()
        self.rel_attr_align_refined_dict = dict()

        self.refined_tuple_candidate_dict = dict()
        self.refined_tuple_dict = dict()

        self.__init()

    def __init(self):
        # if self.PARIS_config is None:
        #     self.PARIS_config = PARISConfig()

        for lite_l in self.kg_l.literal_set:
            if self.kg_r.literal_dict_by_value.__contains__(lite_l.value):
                lite_r = self.kg_r.literal_dict_by_value[lite_l.value]
                self.ent_lite_align_refined_dict[lite_l] = set()
                self.ent_lite_align_refined_dict[lite_l].add(lite_r)
                self.refined_tuple_dict[(lite_l, lite_r)] = 1.0
                self.refined_tuple_dict[(lite_r, lite_l)] = 1.0

    def run(self):
        print("start...")
        for i in range(self.iteration):
            print(str(i) + "th iteration......")
            if i == 0:
                self.__run_per_iteration(init=True)
            else:
                self.__run_per_iteration()

    def __run_per_iteration(self, init=False):
        visited = set()
        print("entity and literal alignment...")
        for (obj_l, obj_r_set) in self.ent_lite_align_refined_dict.items():
            for obj_r in obj_r_set:
                for obj_l_neighbor in obj_l.neighbor_set_inv:
                    for obj_r_neighbor in obj_r.neighbor_set_inv:
                        if obj_l_neighbor.get_type() != obj_r_neighbor.get_type():
                            continue
                        if (obj_l_neighbor, obj_r_neighbor) in visited:
                            continue
                        p_lr = self.__ent_or_lite_align_prob(obj_l_neighbor, obj_r_neighbor, init)
                        if p_lr >= self.threshold:
                            if self.ent_lite_align_refined_dict.__contains__(obj_l_neighbor) is False:
                                self.ent_lite_align_refined_dict[obj_l_neighbor] = set()
                            self.ent_lite_align_candidate_dict[obj_l_neighbor].add(obj_r_neighbor)
                            self.refined_tuple_candidate_dict[(obj_l_neighbor, obj_r_neighbor)] = p_lr
                            self.refined_tuple_candidate_dict[(obj_r_neighbor, obj_l_neighbor)] = p_lr
                        visited.add((obj_l_neighbor, obj_r_neighbor))
                        visited.add((obj_r_neighbor, obj_l_neighbor))
        print("relation alignment...")
        for relation_l in self.kg_l.relation_set:
            for relation_r in self.kg_r.relation_set:
                prob_lr = self.__rel_or_attr_align_prob(relation_l, relation_r)
                prob_rl = self.__rel_or_attr_align_prob(relation_r, relation_l)
                if prob_lr >= self.threshold or prob_rl >= self.threshold:
                    self.refined_tuple_candidate_dict[(relation_l, relation_r)] = prob_lr
                    self.refined_tuple_candidate_dict[(relation_r, relation_l)] = prob_rl
                    print(str(prob_lr) + "\t" + str(prob_rl))
                    if self.rel_attr_align_candidate_dict.__contains__(relation_l) is False:
                        self.rel_attr_align_candidate_dict[relation_l] = set()
                    self.rel_attr_align_candidate_dict[relation_l].add(relation_r)
        print("attribute alignment...")
        for attribute_l in self.kg_l.attribute_set:
            for attribute_r in self.kg_r.attribute_set:
                prob_lr = self.__rel_or_attr_align_prob(attribute_l, attribute_r)
                prob_rl = self.__rel_or_attr_align_prob(attribute_r, attribute_l)
                if prob_lr >= self.threshold or prob_rl >= self.threshold:
                    self.refined_tuple_candidate_dict[(attribute_l, attribute_r)] = prob_lr
                    self.refined_tuple_candidate_dict[(attribute_r, attribute_l)] = prob_rl
                    if self.rel_attr_align_candidate_dict.__contains__(attribute_l) is False:
                        self.rel_attr_align_candidate_dict[attribute_l] = set()
                    if self.rel_attr_align_candidate_dict.__contains__(attribute_r) is False:
                        self.rel_attr_align_candidate_dict[attribute_r] = set()
                    self.rel_attr_align_candidate_dict[attribute_l].add(attribute_r)
                    self.rel_attr_align_candidate_dict[attribute_r].add(attribute_l)
        print("refining...")
        self.__refine_candidate()
        return

    def __refine_candidate(self):
        ent_lite_align_dict = dict()
        refined_tuple_dict = dict()

        for (obj_l, obj_r_set) in self.ent_lite_align_candidate_dict.items():
            candidate_dict = dict()
            for obj_r in obj_r_set:
                candidate_dict[obj_r] = self.refined_tuple_candidate_dict[(obj_l, obj_r)]
            sorted(candidate_dict, key=lambda x: x[1], reverse=True)
            num = 0
            for (candidate, prob) in candidate_dict.items():
                if num == 0:
                    ent_lite_align_dict[obj_l] = set()
                ent_lite_align_dict[obj_l].add(candidate)
                refined_tuple_dict[(obj_l, candidate)] = prob
                refined_tuple_dict[(candidate, obj_l)] = prob
                num += 1
                if num >= self.ent_lite_candidate_num:
                    break

        rel_attr_align_dict = dict()

        for (obj_l, obj_r_set) in self.rel_attr_align_candidate_dict.items():
            candidate_dict = dict()
            for obj_r in obj_r_set:
                candidate_dict[obj_r] = self.refined_tuple_candidate_dict[(obj_l, obj_r)]
            sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)
            num = 0
            for (candidate, prob) in candidate_dict.items():
                if num == 0:
                    rel_attr_align_dict[obj_l] = set()
                rel_attr_align_dict[obj_l].add(candidate)
                refined_tuple_dict[(obj_l, candidate)] = prob
                num += 1
                if num >= self.rel_attr_candidate_num:
                    break
        self.ent_lite_align_refined_dict = ent_lite_align_dict
        self.rel_attr_align_refined_dict = rel_attr_align_dict
        self.refined_tuple_dict = refined_tuple_dict
        self.ent_lite_align_candidate_dict = dict()
        self.rel_attr_align_candidate_dict = dict()
        self.refined_tuple_candidate_dict = dict()
        return

    def __ent_or_lite_align_prob(self, obj_l, obj_r, init=False):
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
                        equality = self.__get_align_prob(tail_l, tail_r)
                        p_lr = self.__get_align_prob(relation_l, relation_r) if init is False else self.theta
                        p_rl = self.__get_align_prob(relation_r, relation_l) if init is False else self.theta
                        prob *= (1.0 - p_lr * relation_r.functionality_inv * equality) * (1.0 - p_rl * relation_l.functionality_inv * equality)
        return 1.0 - prob

    def __rel_or_attr_align_prob(self, obj_l, obj_r):
        numerator, denominator = 0.0, 0.0
        MAX_NUM = 300
        for (head_l, tail_l) in obj_l.tuple_set:
            num = 1.0
            for (head_r, tail_r) in obj_r.tuple_set:
                if self.refined_tuple_dict.__contains__((head_l, head_r)) is False or self.refined_tuple_dict.__contains__((tail_l, tail_r)) is False:
                    continue
                num *= 1.0 - self.__get_align_prob(head_l, head_r) * self.__get_align_prob(tail_l, tail_r)
            numerator += 1.0 - num
            MAX_NUM -= 1
            if MAX_NUM <= 0:
                break

        MAX_NUM = 300
        for (head_l, tail_l) in obj_l.tuple_set:
            num = 1.0
            for head in self.ent_lite_align_refined_dict.get(head_l, set()):
                for tail in self.ent_lite_align_refined_dict.get(tail_l, set()):
                    num *= 1.0 - self.__get_align_prob(head_l, head) * self.__get_align_prob(tail_l, tail)
                denominator += 1.0 - num
            MAX_NUM -= 1
            if MAX_NUM <= 0:
                break
        return numerator / denominator if denominator > 0.0 else 0.0

    def __get_align_prob(self, obj_l, obj_r):
        return self.refined_tuple_dict.get((obj_l, obj_r), 0.0)

    def print_alignment_result(self):
        lite_result_dict = dict()
        ent_result_dict = dict()
        for (obj, counterparts) in self.ent_lite_align_refined_dict.items():
            counterpart, prob = None, None
            for candidate in counterparts:
                if counterpart is None:
                    counterpart = candidate
                    prob = self.refined_tuple_dict[(obj, counterpart)]
                else:
                    if self.refined_tuple_dict[(obj, candidate)] > prob:
                        counterpart = candidate
                        prob = self.refined_tuple_dict[(obj, counterpart)]
            if counterpart is not None:
                if obj.get_type == "ENTITY":
                    ent_result_dict[obj] = counterpart
                else:
                    lite_result_dict[obj] = counterpart

        print("\nAlignment Result:")
        print("--- Literal Alignment ---")
        for (kg_l_lite, kg_r_lite) in lite_result_dict.items():
            print(kg_l_lite.name + "\t" + kg_r_lite.name + "\t" +
                  str(self.refined_tuple_dict[(kg_l_lite, kg_r_lite)]))
        print("--- Entity Alignment ---")
        for (kg_l_ent, kg_r_ent) in ent_result_dict.items():
            print(kg_l_ent.name + "\t" + kg_r_ent.name + "\t" +
                  str(self.refined_tuple_dict[(kg_l_ent, kg_r_ent)]))



