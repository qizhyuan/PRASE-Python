import random
from objects.KG import KG


class KGs:
    def __init__(self, kg1: KG, kg2: KG):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = 0.1
        self.threshold = 0.1
        self.ent_lite_candidate_num = 5
        self.rel_attr_candidate_num = 3
        self.rel_attr_max_sample_num = 10
        self.iteration = 5

        self.ent_lite_align_candidate_dict = dict()
        self.rel_attr_align_candidate_dict = dict()

        self.ent_lite_align_refined_dict = dict()
        self.rel_attr_align_refined_dict = dict()

        self.refined_tuple_candidate_dict = dict()
        self.refined_tuple_dict = dict()

        self.ent_lite_align_set = set()

        self.similarity_func = None

        self._denominator = dict()

        self.__init()

    def __init(self):
        def similarity_func(s1: str, s2: str):
            if s1 == s2:
                return 1.0
            Levenshtein_distance = 0
            s_short, s_long = s1, s2
            if len(s_short) > len(s_long):
                s_short, s_long = s2, s1
            if len(s_short) == 0:
                Levenshtein_distance = len(s_long)
            else:
                def get_element(matrix, idx1, idx2):
                    if idx1 <= 0 and idx2 <= 0:
                        return 0
                    if idx1 <= 0:
                        return idx2
                    if idx2 <= 0:
                        return idx1
                    return matrix[idx1][idx2 % 2]

                dp = [[0] * 2 for _ in range(len(s_short) + 1)]
                for j in range(1, len(s_long) + 1):
                    for i in range(1, len(s_short) + 1):
                        dp[i][j % 2] = min(get_element(dp, i, j - 1) + 1, get_element(dp, i - 1, j) + 1,
                                           get_element(dp, i - 1, j - 1) + (
                                               0 if s_short[i - 1] == s_long[j - 1] else 1))
                Levenshtein_distance = get_element(dp, len(s_short), len(s_long))

            similarity_score = 1.0 - (float(Levenshtein_distance) / max(len(s_short), len(s_long)))
            return similarity_score

        if self.similarity_func is None:
            self.similarity_func = similarity_func

        for lite_l in self.kg_l.literal_set:
            if self.kg_r.literal_dict_by_value.__contains__(lite_l.value):
                lite_r = self.kg_r.literal_dict_by_value[lite_l.value]
                self.ent_lite_align_refined_dict[lite_l] = dict()
                self.ent_lite_align_refined_dict[lite_l][lite_r] = 1.0
                self.refined_tuple_dict[(lite_l, lite_r)] = 1.0
                self.refined_tuple_dict[(lite_r, lite_l)] = 1.0
                self.ent_lite_align_set.add(lite_l)

    def __run_per_iteration(self, init=False):
        visited = set()
        print("entity and literal alignment...")
        for (obj_l, obj_r_dict) in self.ent_lite_align_refined_dict.items():
            for (obj_r, _) in obj_r_dict.items():
                for obj_l_neighbor in obj_l.neighbored_as_tail:
                    for obj_r_neighbor in obj_r.neighbored_as_tail:
                        if obj_l_neighbor.get_type() != obj_r_neighbor.get_type():
                            continue
                        if (obj_l_neighbor, obj_r_neighbor) in visited:
                            continue
                        p_lr = self.__ent_or_lite_align_prob(obj_l_neighbor, obj_r_neighbor, init)
                        # print(obj_l_neighbor.name + "\t" + obj_r_neighbor.name + "\t" + str(p_lr))
                        if p_lr >= self.threshold:
                            if self.ent_lite_align_candidate_dict.__contains__(obj_l_neighbor) is False:
                                self.ent_lite_align_candidate_dict[obj_l_neighbor] = dict()
                            self.ent_lite_align_candidate_dict[obj_l_neighbor][obj_r_neighbor] = p_lr
                            self.refined_tuple_candidate_dict[(obj_l_neighbor, obj_r_neighbor)] = p_lr
                            self.refined_tuple_candidate_dict[(obj_r_neighbor, obj_l_neighbor)] = p_lr
                        visited.add((obj_l_neighbor, obj_r_neighbor))
                        visited.add((obj_r_neighbor, obj_l_neighbor))
                prob = self.__ent_or_lite_align_prob(obj_l, obj_r, init)
                # print(obj_l.name + "\t" + obj_r.name + "\t" + str(prob))
                if prob >= self.threshold:
                    if self.ent_lite_align_candidate_dict.__contains__(obj_l) is False:
                        self.ent_lite_align_candidate_dict[obj_l] = dict()
                    self.ent_lite_align_candidate_dict[obj_l][obj_r] = prob
                    self.refined_tuple_candidate_dict[(obj_l, obj_r)] = prob
                    self.refined_tuple_candidate_dict[(obj_r, obj_l)] = prob
                visited.add((obj_l, obj_r))
                visited.add((obj_r, obj_l))

        self.__refine_ent_lite_candidate()

        print("relation and attribute alignment...")
        # self.__rel_attr_align_per_iteration()

        for obj_l_rel in self.kg_l.relation_set:
            for obj_r_rel in self.kg_r.relation_set:
                if (obj_l_rel, obj_r_rel) in visited:
                    continue
                prob_lr = self.__rel_or_attr_align_prob(obj_l_rel, obj_r_rel)
                prob_rl = self.__rel_or_attr_align_prob(obj_r_rel, obj_l_rel)
                if prob_lr >= self.threshold or prob_rl >= self.threshold:
                    print(obj_l_rel.name + "\t" + obj_r_rel.name + "\t" + str(prob_lr))
                    self.refined_tuple_candidate_dict[(obj_l_rel, obj_r_rel)] = prob_lr
                    self.refined_tuple_candidate_dict[(obj_r_rel, obj_l_rel)] = prob_rl
                    if self.rel_attr_align_candidate_dict.__contains__(obj_l_rel) is False:
                        self.rel_attr_align_candidate_dict[obj_l_rel] = dict()
                    self.rel_attr_align_candidate_dict[obj_l_rel][obj_r_rel] = prob_lr
                visited.add((obj_l_rel, obj_r_rel))

        for obj_l_attr in self.kg_l.attribute_set:
            for obj_r_attr in self.kg_r.attribute_set:
                if (obj_l_attr, obj_r_attr) in visited:
                    continue
                prob_lr = self.__rel_or_attr_align_prob(obj_l_attr, obj_r_attr)
                prob_rl = self.__rel_or_attr_align_prob(obj_r_attr, obj_l_attr)
                if prob_lr >= self.threshold or prob_rl >= self.threshold:
                    print(obj_l_attr.name + "\t" + obj_r_attr.name + "\t" + str(prob_lr))
                    self.refined_tuple_candidate_dict[(obj_l_attr, obj_r_attr)] = prob_lr
                    self.refined_tuple_candidate_dict[(obj_r_attr, obj_l_attr)] = prob_rl
                    if self.rel_attr_align_candidate_dict.__contains__(obj_l_attr) is False:
                        self.rel_attr_align_candidate_dict[obj_l_attr] = dict()
                    self.rel_attr_align_candidate_dict[obj_l_attr][obj_r_attr] = prob_lr
                visited.add((obj_l_attr, obj_r_attr))
        print("refining...")
        self.__refine_rel_attr_candidate()
        self.__clear_candidate_dict()
        print("complete an iteration!")
        return

    def __refine_ent_lite_candidate(self):
        self.ent_lite_align_set = set()
        ent_lite_align_dict, refined_tuple_dict = dict(), dict()
        for (obj_l, obj_r_dict) in self.ent_lite_align_candidate_dict.items():
            sorted(obj_r_dict.items(), key=lambda x: x[1], reverse=True)
            refined_dict = dict()
            num = 0
            for (candidate, prob) in obj_r_dict.items():
                refined_dict[candidate] = prob
                refined_tuple_dict[(obj_l, candidate)] = prob
                refined_tuple_dict[(candidate, obj_l)] = prob
                num += 1
                if num >= self.ent_lite_candidate_num:
                    break
            ent_lite_align_dict[obj_l] = refined_dict
            self.ent_lite_align_set.add(obj_l)

        for (obj_l, obj_r_dict) in self.rel_attr_align_refined_dict.items():
            for (obj_r, prob) in obj_r_dict.items():
                refined_tuple_dict[(obj_l, obj_r)] = prob
        self.ent_lite_align_refined_dict = ent_lite_align_dict
        self.refined_tuple_dict = refined_tuple_dict

    def __rel_attr_align_per_iteration(self):
        for relation_l in self.kg_l.relation_set:
            candidate_dict = dict()
            for head in self.ent_lite_align_set & relation_l.head_ent_set:
                for (candidate, prob) in self.ent_lite_align_refined_dict[head].items():
                    for (relation_candidate, _) in candidate.involved_rel_dict.items():
                        if candidate_dict.__contains__(relation_candidate) is False:
                            candidate_dict[relation_candidate] = 0.0
                        candidate_dict[relation_candidate] += prob

            for tail in self.ent_lite_align_set & relation_l.tail_ent_set:
                for (candidate, prob) in self.ent_lite_align_refined_dict[tail].items():
                    for (relation_candidate, _) in candidate.involved_rel_inv_dict.items():
                        if candidate_dict.__contains__(relation_candidate) is False:
                            candidate_dict[relation_candidate] = 0.0
                        candidate_dict[relation_candidate] += prob
            sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)

            iter_num = 5
            for (relation_r, _) in candidate_dict.items():
                prob_lr = self.__rel_or_attr_align_prob(relation_l, relation_r)
                prob_rl = self.__rel_or_attr_align_prob(relation_r, relation_l)
                if prob_lr >= self.threshold or prob_rl >= self.threshold:
                    print(relation_l.name + "\t" + relation_r.name + "\t" + str(prob_lr))
                    self.refined_tuple_candidate_dict[(relation_l, relation_r)] = prob_lr
                    self.refined_tuple_candidate_dict[(relation_r, relation_l)] = prob_rl
                    if self.rel_attr_align_candidate_dict.__contains__(relation_r) is False:
                        self.rel_attr_align_candidate_dict[relation_l] = dict()
                    self.rel_attr_align_candidate_dict[relation_l][relation_r] = prob_lr
                iter_num -= 1
                if iter_num <= 0:
                    break

        for attribute_l in self.kg_l.attribute_set:
            candidate_dict = dict()
            for entity in self.ent_lite_align_set & attribute_l.entity_set:
                for (candidate, prob) in self.ent_lite_align_refined_dict[entity].items():
                    for (attribute_candidate, _) in candidate.involved_attr_dict.items():
                        if candidate_dict.__contains__(attribute_candidate) is False:
                            candidate_dict[attribute_candidate] = 0.0
                        candidate_dict[attribute_candidate] += prob

            for literal in self.ent_lite_align_set & attribute_l.literal_set:
                for (candidate, prob) in self.ent_lite_align_refined_dict[literal].items():
                    for (attribute_candidate, _) in candidate.involved_as_tail_dict.items():
                        if candidate_dict.__contains__(attribute_candidate) is False:
                            candidate_dict[attribute_candidate] = 0.0
                        candidate_dict[attribute_candidate] += prob
            sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)

            iter_num = 5
            for (attribute_r, _) in candidate_dict.items():
                prob_lr = self.__rel_or_attr_align_prob(attribute_l, attribute_r)
                prob_rl = self.__rel_or_attr_align_prob(attribute_r, attribute_l)
                if prob_lr >= self.threshold or prob_rl >= self.threshold:
                    print(attribute_l.name + "\t" + attribute_r.name + "\t" + str(prob_lr))
                    self.refined_tuple_candidate_dict[(attribute_l, attribute_r)] = prob_lr
                    self.refined_tuple_candidate_dict[(attribute_r, attribute_l)] = prob_rl
                    if self.rel_attr_align_candidate_dict.__contains__(attribute_r) is False:
                        self.rel_attr_align_candidate_dict[attribute_l] = dict()
                    self.rel_attr_align_candidate_dict[attribute_l][attribute_r] = prob_lr
                iter_num -= 1
                if iter_num <= 0:
                    break

    def __refine_rel_attr_candidate(self):
        rel_attr_align_dict, refined_tuple_dict = dict(), dict()

        for (obj_l, obj_r_dict) in self.rel_attr_align_candidate_dict.items():
            candidate_dict = dict()
            sorted(obj_r_dict.items(), key=lambda x: x[1], reverse=True)
            num = 0
            for (candidate, prob) in obj_r_dict.items():
                candidate_dict[candidate] = prob
                refined_tuple_dict[(obj_l, candidate)] = prob
                num += 1
                if num >= self.rel_attr_candidate_num:
                    break
            rel_attr_align_dict[obj_l] = candidate_dict

        for (obj_l, obj_r_dict) in self.ent_lite_align_refined_dict.items():
            for (obj_r, prob) in obj_r_dict.items():
                refined_tuple_dict[(obj_l, obj_r)] = prob
        self.rel_attr_align_refined_dict = rel_attr_align_dict
        self.refined_tuple_dict = refined_tuple_dict
        return

    def __clear_candidate_dict(self):
        self.ent_lite_align_candidate_dict = dict()
        self.rel_attr_align_candidate_dict = dict()
        self.refined_tuple_candidate_dict = dict()
        self._denominator = dict()

    def __ent_or_lite_align_prob(self, obj_l, obj_r, init=False):
        prob = 1.0
        if obj_l.get_type() != obj_r.get_type():
            return 0.0
        if obj_l.get_type() == "LITERAL":
            return self.similarity_func(obj_l.value, obj_r.value)
        for (rel_l, tail_set_l) in obj_l.involved_as_head_dict.items():
            for (rel_r, tail_set_r) in obj_r.involved_as_head_dict.items():
                if init is False and self.refined_tuple_dict.__contains__((rel_l, rel_r)) is False and \
                        self.refined_tuple_dict.__contains__((rel_r, rel_l)) is False:
                    continue
                for tail_l in tail_set_l:
                    for tail_r in tail_set_r:
                        if self.refined_tuple_dict.__contains__((tail_l, tail_r)) is False:
                            continue
                        equality = self.__get_align_prob(tail_l, tail_r)
                        if equality <= 0.01:
                            continue
                        p_lr = self.__get_align_prob(rel_l, rel_r) if init is False else self.theta
                        p_rl = self.__get_align_prob(rel_r, rel_l) if init is False else self.theta
                        prob *= (1.0 - p_lr * rel_r.functionality_inv * equality) * (
                                    1.0 - p_rl * rel_l.functionality_inv * equality)

        return 1.0 - prob

    def __rel_or_attr_align_prob(self, obj_l, obj_r):
        numerator, denominator = 0.0, 0.0

        for (head_l, tail_l) in obj_l.tuple_set:
            num = 1.0
            for head_r in self.ent_lite_align_refined_dict.get(head_l, set()):
                for tail_r in self.ent_lite_align_refined_dict.get(tail_l, set()):
                    if (head_r, tail_r) in obj_r.tuple_set:
                        num *= 1.0 - self.__get_align_prob(head_l, head_r) * self.__get_align_prob(tail_l, tail_r)
            numerator += 1.0 - num

        if self._denominator.__contains__(obj_l):
            denominator = self._denominator[obj_l]
        else:
            for (head_l, tail_l) in obj_l.tuple_set:
                num = 1.0
                # print(self.ent_lite_align_refined_dict)
                for (head, _) in self.ent_lite_align_refined_dict.get(head_l, dict()).items():
                    for (tail, _) in self.ent_lite_align_refined_dict.get(tail_l, dict()).items():
                        num *= 1.0 - self.__get_align_prob(head_l, head) * self.__get_align_prob(tail_l, tail)
                    denominator += 1.0 - num
            self._denominator[obj_l] = denominator
        return numerator / denominator if denominator > 0.0 else 0.0

    def __get_align_prob(self, obj_l, obj_r):
        return self.refined_tuple_dict.get((obj_l, obj_r), 0.0)

    def __dict_result_handler(self, refined_dict, obj_type: str):
        result_dict_first, result_dict_second = dict(), dict()
        for (obj, counterpart_dict) in refined_dict.items():
            counterpart, prob_max = None, None
            for (candidate, prob) in counterpart_dict.items():
                if counterpart is None:
                    counterpart = candidate
                    prob_max = prob
                if prob > prob_max:
                    counterpart = candidate
            if counterpart is not None:
                if obj.get_type() == obj_type:
                    result_dict_first[obj] = counterpart
                else:
                    result_dict_second[obj] = counterpart
        return result_dict_first, result_dict_second

    def __result_writer(self, f, result_dict, title):
        f.write("--- " + title + " ---\n")
        for (obj_l, obj_r) in result_dict.items():
            f.write(obj_l.name + "\t" + obj_r.name + "\t" + str(
                self.refined_tuple_dict[(obj_l, obj_r)]) + "\n")

    def run(self):
        print("start...")
        for i in range(self.iteration):
            print(str(i) + "-th iteration......")
            if i == 0:
                self.__run_per_iteration(init=True)
            else:
                self.__run_per_iteration()

    def store_params(self):
        return

    def load_params(self):
        return

    def output_alignment_result(self, path="alignment_result.txt"):
        ent_result_dict, lite_result_dict = self.__dict_result_handler(self.ent_lite_align_refined_dict, "ENTITY")
        rel_result_dict, attr_result_dict = self.__dict_result_handler(self.rel_attr_align_refined_dict, "RELATION")

        with open(path, "w+", encoding="utf8") as f:
            f.write("Alignment Result:\n")
            self.__result_writer(f, attr_result_dict, "Attribute Alignment")
            self.__result_writer(f, rel_result_dict, "Relation Alignment")
            self.__result_writer(f, lite_result_dict, "Literal Alignment")
            self.__result_writer(f, ent_result_dict, "Entity Alignment")