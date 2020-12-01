from objects.KG import KG


class KGs:
    def __init__(self, kg1: KG, kg2: KG):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = 0.1
        self.threshold = 0.1
        self.ent_lite_candidate_num = 5
        self.rel_attr_candidate_num = 3
        self.iteration = 3

        self.ent_lite_align_candidate_dict = dict()
        self.rel_attr_align_candidate_dict = dict()

        self.ent_lite_align_refined_dict = dict()
        self.rel_attr_align_refined_dict = dict()

        self.refined_tuple_candidate_dict = dict()
        self.refined_tuple_dict = dict()

        self.__init()

    def __init(self):
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
            print(str(i) + "-th iteration......")
            if i == 0:
                self.__run_per_iteration(init=True)
            else:
                self.__run_per_iteration()

    def __run_per_iteration(self, init=False):
        visited = set()
        print("entity and literal alignment...")
        for (obj_l, obj_r_set) in self.ent_lite_align_refined_dict.items():
            for obj_r in obj_r_set:
                for obj_l_neighbor in obj_l.neighbored_as_tail:
                    for obj_r_neighbor in obj_r.neighbored_as_tail:
                        if obj_l_neighbor.get_type() != obj_r_neighbor.get_type():
                            continue
                        if (obj_l_neighbor, obj_r_neighbor) in visited:
                            continue
                        p_lr = self.__ent_or_lite_align_prob(obj_l_neighbor, obj_r_neighbor, init)
                        print(obj_l_neighbor.name + "\t" + obj_r_neighbor.name + "\t" + str(p_lr))
                        if p_lr >= self.threshold:
                            if self.ent_lite_align_candidate_dict.__contains__(obj_l_neighbor) is False:
                                self.ent_lite_align_candidate_dict[obj_l_neighbor] = set()
                            self.ent_lite_align_candidate_dict[obj_l_neighbor].add(obj_r_neighbor)
                            self.refined_tuple_candidate_dict[(obj_l_neighbor, obj_r_neighbor)] = p_lr
                            self.refined_tuple_candidate_dict[(obj_r_neighbor, obj_l_neighbor)] = p_lr
                        visited.add((obj_l_neighbor, obj_r_neighbor))
                        visited.add((obj_r_neighbor, obj_l_neighbor))
                prob = self.__ent_or_lite_align_prob(obj_l, obj_r, init)
                print(obj_l.name + "\t" + obj_r.name + "\t" + str(prob))
                if prob >= self.threshold:
                    if self.ent_lite_align_candidate_dict.__contains__(obj_l) is False:
                        self.ent_lite_align_candidate_dict[obj_l] = set()
                    self.ent_lite_align_candidate_dict[obj_l].add(obj_r)
                    self.refined_tuple_candidate_dict[(obj_l, obj_r)] = prob
                    self.refined_tuple_candidate_dict[(obj_r, obj_l)] = prob
                visited.add((obj_l, obj_r))
                visited.add((obj_r, obj_l))

        print("relation and attribute alignment...")
        for (obj_l, obj_r_set) in self.ent_lite_align_refined_dict.items():
            for obj_r in obj_r_set:
                for (obj_l_rel, _) in obj_l.involved_as_tail_dict.items():
                    for (obj_r_rel, _) in obj_r.involved_as_tail_dict.items():
                        if obj_l_rel.get_type() != obj_r_rel.get_type():
                            continue
                        if (obj_l_rel, obj_r_rel) in visited:
                            continue
                        prob_lr = self.__rel_or_attr_align_prob(obj_l_rel, obj_r_rel)
                        prob_rl = self.__rel_or_attr_align_prob(obj_r_rel, obj_l_rel)
                        if prob_lr >= self.threshold or prob_rl >= self.threshold:
                            self.refined_tuple_candidate_dict[(obj_l_rel, obj_r_rel)] = prob_lr
                            self.refined_tuple_candidate_dict[(obj_r_rel, obj_l_rel)] = prob_rl
                            if self.rel_attr_align_candidate_dict.__contains__(obj_l_rel) is False:
                                self.rel_attr_align_candidate_dict[obj_l_rel] = set()
                            self.rel_attr_align_candidate_dict[obj_l_rel].add(obj_r_rel)
                        visited.add((obj_l_rel, obj_r_rel))
        print("refining...")
        self.__refine_candidate()
        print("complete an iteration!")
        return

    def __refine_candidate(self):
        ent_lite_align_dict = dict()
        refined_tuple_dict = dict()

        for (obj_l, obj_r_set) in self.ent_lite_align_candidate_dict.items():
            candidate_dict = dict()
            for obj_r in obj_r_set:
                candidate_dict[obj_r] = self.refined_tuple_candidate_dict[(obj_l, obj_r)]
            sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)
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
        if obj_l.get_type() != obj_r.get_type():
            return 0.0
        if obj_l.get_type() == "LITERAL":
            return self.__get_similarity(obj_l.value, obj_r.value)
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
                        p_lr = self.__get_align_prob(rel_l, rel_r) if init is False else self.theta
                        p_rl = self.__get_align_prob(rel_r, rel_l) if init is False else self.theta
                        prob *= (1.0 - p_lr * rel_r.functionality_inv * equality) * (
                                    1.0 - p_rl * rel_l.functionality_inv * equality)

        return 1.0 - prob

    def __rel_or_attr_align_prob(self, obj_l, obj_r):
        numerator, denominator = 0.0, 0.0
        MAX_NUM = 10000
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

        MAX_NUM = 10000
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

    def output_alignment_result(self, path="alignment_result.txt"):
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
                if obj.get_type() == "ENTITY":
                    ent_result_dict[obj] = counterpart
                else:
                    lite_result_dict[obj] = counterpart
        with open(path, "w+", encoding="utf8") as f:
            f.write("Alignment Result:\n")
            f.write("--- Literal Alignment ---\n")
            for (kg_l_lite, kg_r_lite) in lite_result_dict.items():
                f.write(kg_l_lite.name + "\t" + kg_r_lite.name + "\t" + str(self.refined_tuple_dict[(kg_l_lite, kg_r_lite)]) + "\n")
            f.write("--- Entity Alignment ---\n")
            for (kg_l_ent, kg_r_ent) in ent_result_dict.items():
                f.write(kg_l_ent.name + "\t" + kg_r_ent.name + "\t" + str(self.refined_tuple_dict[(kg_l_ent, kg_r_ent)]) + "\n")

    def __get_similarity(self, s1: str, s2: str):
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


