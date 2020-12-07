import os
from objects.KG import KG


class KGs:
    def __init__(self, kg1: KG, kg2: KG, ent_lite_candidate_num=1, rel_attr_candidate_num=1, theta=0.1,
                 refine_threshold=0.1, output_threshold=0.9, iteration=3):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = theta
        self.refine_threshold = refine_threshold
        self.output_threshold = output_threshold
        self.ent_lite_candidate_num = ent_lite_candidate_num
        self.rel_attr_candidate_num = rel_attr_candidate_num
        self.iteration = iteration

        self.ent_lite_align_candidate_dict = dict()
        self.rel_attr_align_candidate_dict = dict()

        self.ent_lite_align_refined_dict = dict()
        self.rel_attr_align_refined_dict = dict()

        self.refined_tuple_candidate_dict = dict()
        self.refined_tuple_dict = dict()

        # self.ent_lite_align_set = set()

        self.similarity_func = None

        self._rel_or_attr_align_prob_denominator = dict()

        self.__init()

    def __init(self):
        def similarity_func(s1: str, s2: str):
            if s1 == s2:
                return 1.0
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
                if self.ent_lite_align_refined_dict.__contains__(lite_r) is False:
                    self.ent_lite_align_refined_dict[lite_r] = dict()
                self.ent_lite_align_refined_dict[lite_r][lite_l] = 1.0
                self.refined_tuple_dict[(lite_l, lite_r)] = 1.0
                self.refined_tuple_dict[(lite_r, lite_l)] = 1.0
                # print(lite_l.name + "\t" + lite_r.name)
                # self.ent_lite_align_set.add(lite_l)

    def __run_per_iteration(self, init=False):

        print("Entity and Literal Alignment...")
        self.__ent_lite_align_per_iteration(init)

        print("Refining...")
        self.__refine_ent_lite_candidate()

        print("Relation and Attribute Alignment...")
        self.__rel_attr_align_per_iteration()

        print("Refining...")
        self.__refine_rel_attr_candidate()
        self.__clear_candidate_dict()
        print("Complete an Iteration!")
        return

    def __refine_ent_lite_candidate(self):
        # self.ent_lite_align_set = set()
        ent_lite_align_dict, refined_tuple_dict = dict(), dict()
        for (obj_l, obj_r_dict) in self.ent_lite_align_candidate_dict.items():
            sorted(obj_r_dict.items(), key=lambda x: x[1], reverse=True)
            refined_dict = dict()
            candidate_num = 0
            for (candidate, prob) in obj_r_dict.items():
                refined_dict[candidate] = prob
                refined_tuple_dict[(obj_l, candidate)] = prob
                refined_tuple_dict[(candidate, obj_l)] = prob
                candidate_num += 1
                if candidate_num >= self.ent_lite_candidate_num:
                    break
            ent_lite_align_dict[obj_l] = refined_dict
            # self.ent_lite_align_set.add(obj_l)

        for (obj_l, obj_r_dict) in self.rel_attr_align_refined_dict.items():
            for (obj_r, prob) in obj_r_dict.items():
                refined_tuple_dict[(obj_l, obj_r)] = prob
        self.ent_lite_align_refined_dict = ent_lite_align_dict
        self.refined_tuple_dict = refined_tuple_dict

    def __ent_lite_align_per_iteration(self, init=False):
        visited = set()
        for (obj_l, obj_r_dict) in self.ent_lite_align_refined_dict.items():
            if obj_l.affiliation is self.kg_r:
                continue
            for (obj_r, _) in obj_r_dict.items():
                for obj_l_neighbor in obj_l.neighbored_as_tail:
                    for obj_r_neighbor in obj_r.neighbored_as_tail:
                        if obj_l_neighbor.get_type() != obj_r_neighbor.get_type():
                            continue
                        if (obj_l_neighbor, obj_r_neighbor) in visited:
                            continue
                        p_lr = self.__ent_or_lite_align_prob(obj_l_neighbor, obj_r_neighbor, init)
                        if p_lr >= self.refine_threshold:
                            if self.ent_lite_align_candidate_dict.__contains__(obj_l_neighbor) is False:
                                self.ent_lite_align_candidate_dict[obj_l_neighbor] = dict()
                            if self.ent_lite_align_candidate_dict.__contains__(obj_r_neighbor) is False:
                                self.ent_lite_align_candidate_dict[obj_r_neighbor] = dict()
                            self.ent_lite_align_candidate_dict[obj_l_neighbor][obj_r_neighbor] = p_lr
                            self.ent_lite_align_candidate_dict[obj_r_neighbor][obj_l_neighbor] = p_lr
                            self.refined_tuple_candidate_dict[(obj_l_neighbor, obj_r_neighbor)] = p_lr
                            self.refined_tuple_candidate_dict[(obj_r_neighbor, obj_l_neighbor)] = p_lr
                        visited.add((obj_l_neighbor, obj_r_neighbor))
                        visited.add((obj_r_neighbor, obj_l_neighbor))
                prob = self.__ent_or_lite_align_prob(obj_l, obj_r, init)
                if prob >= self.refine_threshold:
                    if self.ent_lite_align_candidate_dict.__contains__(obj_l) is False:
                        self.ent_lite_align_candidate_dict[obj_l] = dict()
                    if self.ent_lite_align_candidate_dict.__contains__(obj_r) is False:
                        self.ent_lite_align_candidate_dict[obj_r] = dict()
                    self.ent_lite_align_candidate_dict[obj_l][obj_r] = prob
                    self.ent_lite_align_candidate_dict[obj_r][obj_l] = prob
                    self.refined_tuple_candidate_dict[(obj_l, obj_r)] = prob
                    self.refined_tuple_candidate_dict[(obj_r, obj_l)] = prob
                visited.add((obj_l, obj_r))
                visited.add((obj_r, obj_l))

    def __rel_attr_align_per_iteration(self):
        visited = set()
        for obj_l_rel in self.kg_l.relation_set:
            for obj_r_rel in self.kg_r.relation_set:
                if (obj_l_rel, obj_r_rel) in visited:
                    continue
                prob_lr = self.__rel_or_attr_align_prob(obj_l_rel, obj_r_rel)
                prob_rl = self.__rel_or_attr_align_prob(obj_r_rel, obj_l_rel)
                if prob_lr >= self.refine_threshold or prob_rl >= self.refine_threshold:
                    # print(obj_l_rel.name + "\t" + obj_r_rel.name + "\t" + str(prob_lr) + "\t" + str(prob_rl))
                    self.refined_tuple_candidate_dict[(obj_l_rel, obj_r_rel)] = prob_lr
                    self.refined_tuple_candidate_dict[(obj_r_rel, obj_l_rel)] = prob_rl
                    if self.rel_attr_align_candidate_dict.__contains__(obj_l_rel) is False:
                        self.rel_attr_align_candidate_dict[obj_l_rel] = dict()
                    if self.rel_attr_align_candidate_dict.__contains__(obj_r_rel) is False:
                        self.rel_attr_align_candidate_dict[obj_r_rel] = dict()
                    self.rel_attr_align_candidate_dict[obj_l_rel][obj_r_rel] = prob_lr
                    self.rel_attr_align_candidate_dict[obj_r_rel][obj_l_rel] = prob_rl
                visited.add((obj_l_rel, obj_r_rel))

        for obj_l_attr in self.kg_l.attribute_set:
            for obj_r_attr in self.kg_r.attribute_set:
                if (obj_l_attr, obj_r_attr) in visited:
                    continue
                prob_lr = self.__rel_or_attr_align_prob(obj_l_attr, obj_r_attr)
                prob_rl = self.__rel_or_attr_align_prob(obj_r_attr, obj_l_attr)
                if prob_lr >= self.refine_threshold or prob_rl >= self.refine_threshold:
                    # print(obj_l_attr.name + "\t" + obj_r_attr.name + "\t" + str(prob_lr) + "\t" + str(prob_rl))
                    self.refined_tuple_candidate_dict[(obj_l_attr, obj_r_attr)] = prob_lr
                    self.refined_tuple_candidate_dict[(obj_r_attr, obj_l_attr)] = prob_rl
                    if self.rel_attr_align_candidate_dict.__contains__(obj_l_attr) is False:
                        self.rel_attr_align_candidate_dict[obj_l_attr] = dict()
                    if self.rel_attr_align_candidate_dict.__contains__(obj_r_attr) is False:
                        self.rel_attr_align_candidate_dict[obj_r_attr] = dict()
                    self.rel_attr_align_candidate_dict[obj_l_attr][obj_r_attr] = prob_lr
                    self.rel_attr_align_candidate_dict[obj_r_attr][obj_l_attr] = prob_rl
                visited.add((obj_l_attr, obj_r_attr))

    def __refine_rel_attr_candidate(self):
        rel_attr_align_dict, refined_tuple_dict = dict(), dict()

        for (obj_l, obj_r_dict) in self.rel_attr_align_candidate_dict.items():
            candidate_dict = dict()
            sorted(obj_r_dict.items(), key=lambda x: x[1], reverse=True)
            candidate_num = 0
            for (candidate, prob) in obj_r_dict.items():
                candidate_dict[candidate] = prob
                refined_tuple_dict[(obj_l, candidate)] = prob
                candidate_num += 1
                if candidate_num >= self.rel_attr_candidate_num:
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
        self._rel_or_attr_align_prob_denominator = dict()

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

        if self._rel_or_attr_align_prob_denominator.__contains__(obj_l):
            denominator = self._rel_or_attr_align_prob_denominator[obj_l]
        else:
            for (head_l, tail_l) in obj_l.tuple_set:
                num = 1.0
                for (head, _) in self.ent_lite_align_refined_dict.get(head_l, dict()).items():
                    for (tail, _) in self.ent_lite_align_refined_dict.get(tail_l, dict()).items():
                        num *= 1.0 - self.__get_align_prob(head_l, head) * self.__get_align_prob(tail_l, tail)
                    denominator += 1.0 - num
            self._rel_or_attr_align_prob_denominator[obj_l] = denominator
        return numerator / denominator if denominator > 0.0 else 0.0

    def __get_align_prob(self, obj_l, obj_r):
        return self.refined_tuple_dict.get((obj_l, obj_r), 0.0)

    def __ent_lite_dict_result_handler(self, refined_dict, obj_type: str, threshold=0.0):
        result_dict_first, result_dict_second = dict(), dict()
        for (obj, counterpart_dict) in refined_dict.items():
            if obj.affiliation is self.kg_r:
                continue
            counterpart, prob_max = None, None
            for (candidate, prob) in counterpart_dict.items():
                if counterpart is None:
                    counterpart = candidate
                    prob_max = prob
                if prob > prob_max:
                    counterpart = candidate
            if counterpart is not None:
                if prob_max < threshold:
                    continue
                if obj.get_type() == obj_type:
                    result_dict_first[obj] = counterpart
                else:
                    result_dict_second[obj] = counterpart
        return result_dict_first, result_dict_second

    def __rel_attr_dict_result_handler(self, refined_dict, obj_type: str, threshold=0.0):
        result_dict_first, result_dict_inv_first = dict(), dict()
        result_dict_second, result_dict_inv_second = dict(), dict()
        for (obj, counterpart_dict) in refined_dict.items():
            counterpart, prob_max = None, None
            for (candidate, prob) in counterpart_dict.items():
                if counterpart is None:
                    counterpart = candidate
                    prob_max = prob
                if prob > prob_max:
                    counterpart = candidate
            if counterpart is not None:
                if prob_max < threshold:
                    continue
                if obj.get_type() == obj_type:
                    if obj.affiliation is self.kg_l:
                        result_dict_first[obj] = counterpart
                    else:
                        result_dict_inv_first[obj] = counterpart
                else:
                    if obj.affiliation is self.kg_l:
                        result_dict_second[obj] = counterpart
                    else:
                        result_dict_inv_second[obj] = counterpart

        return result_dict_first, result_dict_second, result_dict_inv_first, result_dict_inv_second

    def __result_writer(self, f, result_dict, title):
        f.write("--- " + title + " ---\n\n")
        for (obj_l, obj_r) in result_dict.items():
            f.write(obj_l.name + "\t" + obj_r.name + "\t" + str(
                self.refined_tuple_dict[(obj_l, obj_r)]) + "\n")
        f.write("\n")

    def run(self):
        print("Start...")
        for i in range(self.iteration):
            print(str(i + 1) + "-th iteration......")
            if i == 0 and len(self.rel_attr_align_refined_dict) == 0:
                self.__run_per_iteration(init=True)
            else:
                self.__run_per_iteration()
            # path_validation = "dataset/industry/ent_links"
            # for i in range(9):
            #     validate_threshold = 0.1 * float(i)
            #     self.validate(path_validation, validate_threshold)
        print("PARIS Completed!")

    @staticmethod
    def __store_params_writer(tuple_set, path):
        with open(path, "w+", encoding="utf8") as f:
            for params in tuple_set:
                f.write('\t'.join([params[0], params[1], str(params[2])]))
                f.write('\n')

    def store_params(self, path="output/"):
        ent_align_tuple_set, lite_align_tuple_set = set(), set()
        rel_align_tuple_set, rel_inv_align_tuple_set = set(), set()
        attr_align_tuple_set, attr_inv_align_tuple_set = set(), set()

        for (refined_tuple, prob) in self.refined_tuple_dict.items():
            param1, param2 = refined_tuple[0], refined_tuple[1]
            if param1.get_type() == "ENTITY" or param1.get_type() == "LITERAL":
                if param1.affiliation is self.kg_r:
                    continue
                else:
                    if param1.get_type() == "ENTITY":
                        ent_align_tuple_set.add((param1.name, param2.name, prob))
                    else:
                        lite_align_tuple_set.add((param1.name, param2.name, prob))
            else:
                if param1.affiliation is self.kg_l:
                    if param1.get_type() == "RELATION":
                        rel_align_tuple_set.add((param1.name, param2.name, prob))
                    else:
                        attr_align_tuple_set.add((param1.name, param2.name, prob))
                else:
                    if param1.get_type() == "RELATION":
                        rel_inv_align_tuple_set.add((param1.name, param2.name, prob))
                    else:
                        attr_inv_align_tuple_set.add((param1.name, param2.name, prob))
        self.__store_params_writer(ent_align_tuple_set, os.path.join(path, "ent_align_tuple_set"))
        self.__store_params_writer(lite_align_tuple_set, os.path.join(path, "lite_align_tuple_set"))
        self.__store_params_writer(rel_align_tuple_set, os.path.join(path, "rel_align_tuple_set"))
        self.__store_params_writer(attr_align_tuple_set, os.path.join(path, "attr_align_tuple_set"))
        self.__store_params_writer(rel_inv_align_tuple_set, os.path.join(path, "rel_inv_align_tuple_set"))
        self.__store_params_writer(attr_inv_align_tuple_set, os.path.join(path, "attr_inv_align_tuple_set"))

    def load_params(self, path="output/"):
        self.__load_params_helper(os.path.join(path, "ent_align_tuple_set"), "ENTITY")
        self.__load_params_helper(os.path.join(path, "lite_align_tuple_set"), "LITERAL")
        self.__load_params_helper(os.path.join(path, "rel_align_tuple_set"), "RELATION")
        self.__load_params_helper(os.path.join(path, "rel_inv_align_tuple_set"), "RELATION", inv=True)
        self.__load_params_helper(os.path.join(path, "attr_align_tuple_set"), "ATTRIBUTE")
        self.__load_params_helper(os.path.join(path, "attr_inv_align_tuple_set"), "ATTRIBUTE", inv=True)
        return

    def __load_params_helper(self, path, data_type, inv=False):
        with open(path, "r", encoding="utf8") as f:
            for line in f.readlines():
                params = str.strip(line).split(sep="\t")
                if len(params) != 3:
                    continue
                obj_l, obj_r, prob = params[0].strip(), params[1].strip(), float(params[2].strip())
                if data_type == "ENTITY":
                    self.insert_ent_tuple(obj_l, obj_r, prob)
                if data_type == "LITERAL":
                    self.insert_lite_tuple(obj_l, obj_r, prob)
                if data_type == "RELATION":
                    self.insert_rel_tuple(obj_l, obj_r, prob, inv)
                if data_type == "ATTRIBUTE":
                    self.insert_attr_tuple(obj_l, obj_r, prob, inv)

    def insert_ent_tuple(self, ent_l: str, ent_r: str, prob=1.0):
        obj_l, obj_r = self.kg_l.entity_dict_by_name.get(ent_l.strip()), self.kg_r.entity_dict_by_name.get(ent_r.strip())
        if obj_l is None:
            print("Exception: fail to load Entity (" + ent_l + ")")
        if obj_r is None:
            print("Exception: fail to load Entity (" + ent_r + ")")
        if obj_l is None or obj_r is None:
            return
        prob = float(prob)
        self.__insert_ent_or_lite_tuple(obj_l, obj_r, prob)

    def insert_lite_tuple(self, lite_l: str, lite_r: str, prob=1.0):
        obj_l, obj_r = self.kg_l.literal_dict_by_name.get(lite_l.strip()), self.kg_r.literal_dict_by_name.get(
            lite_r.strip())
        if obj_l is None:
            print("Exception: fail to load Literal (" + lite_l + ")")
        if obj_r is None:
            print("Exception: fail to load Literal (" + lite_r + ")")
        if obj_l is None or obj_r is None:
            return
        prob = float(prob)
        self.__insert_ent_or_lite_tuple(obj_l, obj_r, prob)

    def insert_rel_tuple(self, rel_l: str, rel_r: str, prob=1.0, inv=False):
        if inv:
            rel_l, rel_r = rel_r, rel_l
        obj_l, obj_r = self.kg_l.relation_dict_by_name.get(rel_l.strip()), self.kg_r.relation_dict_by_name.get(
            rel_r.strip())
        if obj_l is None:
            print("Exception: fail to load Relation (" + rel_l + ")")
        if obj_r is None:
            print("Exception: fail to load Relation (" + rel_r + ")")
        if obj_l is None or obj_r is None:
            return
        prob = float(prob)
        if inv:
            self.__insert_rel_attr_tuple(obj_r, obj_l, prob)
        else:
            self.__insert_rel_attr_tuple(obj_l, obj_r, prob)

    def insert_attr_tuple(self, attr_l: str, attr_r: str, prob=1.0, inv=False):
        if inv:
            attr_l, attr_r = attr_r, attr_l
        obj_l, obj_r = self.kg_l.attribute_dict_by_name.get(attr_l.strip()), self.kg_r.attribute_dict_by_name.get(
            attr_r.strip())
        if obj_l is None:
            print("Exception: fail to load Attribute (" + attr_l + ")")
        if obj_r is None:
            print("Exception: fail to load Attribute (" + attr_r + ")")
        if obj_l is None or obj_r is None:
            return
        prob = float(prob)
        if inv:
            self.__insert_rel_attr_tuple(obj_r, obj_l, prob)
        else:
            self.__insert_rel_attr_tuple(obj_l, obj_r, prob)

    def __insert_ent_or_lite_tuple(self, obj_l, obj_r, prob):
        if self.ent_lite_align_refined_dict.__contains__(obj_l) is False:
            self.ent_lite_align_refined_dict[obj_l] = dict()
        if self.ent_lite_align_refined_dict.__contains__(obj_r) is False:
            self.ent_lite_align_refined_dict[obj_r] = dict()
        self.ent_lite_align_refined_dict[obj_l][obj_r] = prob
        self.ent_lite_align_refined_dict[obj_r][obj_l] = prob
        self.refined_tuple_dict[(obj_l, obj_r)] = prob

    def __insert_rel_attr_tuple(self, obj_l, obj_r, prob):
        if self.rel_attr_align_refined_dict.__contains__(obj_l) is False:
            self.rel_attr_align_refined_dict[obj_l] = dict()
        self.rel_attr_align_refined_dict[obj_l][obj_r] = prob
        self.refined_tuple_dict[(obj_l, obj_r)] = prob

    def store_results(self, path="output/EA_Result.txt"):
        ent_result_dict, lite_result_dict = self.__ent_lite_dict_result_handler(
            self.ent_lite_align_refined_dict, "ENTITY", threshold=self.output_threshold)
        rel_result_dict, attr_result_dict, rel_result_inv_dict, attr_result_inv_dict = \
            self.__rel_attr_dict_result_handler(self.rel_attr_align_refined_dict, "RELATION")

        base, _ = os.path.split(path)
        if not os.path.exists(base):
            os.makedirs(base)

        with open(path, "w+", encoding="utf8") as f:
            f.write("Alignment Result:\n\n")
            self.__result_writer(f, attr_result_dict, "Attribute Alignment")
            self.__result_writer(f, attr_result_inv_dict, "Attribute INV Alignment")
            self.__result_writer(f, rel_result_dict, "Relation Alignment")
            self.__result_writer(f, rel_result_inv_dict, "Relation INV Alignment")
            self.__result_writer(f, lite_result_dict, "Literal Alignment")
            self.__result_writer(f, ent_result_dict, "Entity Alignment")

    def validate(self, path, threshold=None):
        if threshold is None:
            threshold = self.output_threshold
        else:
            threshold = float(threshold)
        correct_num, total_num = 0.0, 0.0
        ent_align_result = set()
        for (obj_l, obj_r_dict) in self.ent_lite_align_refined_dict.items():
            if obj_l.get_type() == "LITERAL" or obj_l.affiliation is self.kg_r:
                continue
            counterpart, prob_max = None, None
            for (candidate, prob) in obj_r_dict.items():
                if counterpart is None:
                    counterpart = candidate
                    prob_max = prob
                if prob > prob_max:
                    counterpart = candidate
            if counterpart is not None:
                if prob_max < threshold:
                    continue
                ent_align_result.add((obj_l, counterpart))
        if len(ent_align_result) == 0:
            print("Exception: no satisfied alignment result with threshold=" + str(threshold))
            return
        with open(path, "r", encoding="utf8") as f:
            for line in f.readlines():
                params = str.strip(line).split("\t")
                assert len(params) == 2
                ent_l, ent_r = params[0].strip(), params[1].strip()
                obj_l, obj_r = self.kg_l.entity_dict_by_name.get(ent_l), self.kg_r.entity_dict_by_name.get(ent_r)
                if obj_l is None:
                    print("Exception: fail to load Entity (" + ent_l + ")")
                if obj_r is None:
                    print("Exception: fail to load Entity (" + ent_r + ")")
                if obj_l is None or obj_r is None:
                    continue
                if (obj_l, obj_r) in ent_align_result:
                    correct_num += 1.0
                total_num += 1.0

        if total_num == 0.0:
            print("Exception: no satisfied instance for validation")
        else:
            precision, recall = correct_num / len(ent_align_result), correct_num / total_num
            print("Precision: " + str(precision) + "\tRecall: " + str(recall))

    # def ent_matcher(self, threshold=None):
    #     if threshold is None:
    #         threshold = self.output_threshold
    #     visited, ent_align_result = set(), set()
    #     sorted(self.refined_tuple_dict.items(), key=lambda x: x[1], reverse=True)
    #     for (obj_tuple, prob) in self.refined_tuple_dict.items():
    #         obj_l, obj_r = obj_tuple[0], obj_tuple[1]
    #         if obj_l.get_type() != "ENTITY":
    #             continue
    #         if obj_l in visited or obj_r in visited:
    #             continue
    #         if prob < threshold:
    #             break
    #         if obj_l.affiliation is self.kg_l:
    #             ent_align_result.add((obj_l, obj_r))
    #         else:
    #             ent_align_result.add((obj_r, obj_l))
    #         visited.add(obj_l)
    #         visited.add(obj_r)
    #     return ent_align_result
