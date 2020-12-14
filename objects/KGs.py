import random
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from objects.KGsUtil import KGsUtil
from objects.KG import KG


class KGs:
    def __init__(self, kg1: KG, kg2: KG, ent_candidate_num=1, rel_candidate_num=1, theta=0.1, iteration=3):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = theta
        self.ent_candidate_num = ent_candidate_num
        self.rel_candidate_num = rel_candidate_num
        self.iteration = iteration
        self.epsilon = 0.01

        self.lite_align_dict = dict()
        self.lite_align_tuple_dict = dict()

        self.rel_align_ongoing_dict = dict()
        self.rel_align_norm_dict = dict()

        self.rel_align_candidate_dict = dict()

        self.ent_align_ongoing_dict = dict()

        self.ent_align_refined_dict = dict()
        self.rel_align_refined_dict = dict()

        self.refined_tuple_candidate_dict = dict()
        self.refined_tuple_dict = dict()

        self._iter_num = 0
        self.util = KGsUtil(self)
        self.__init()

    def __init(self):
        for lite_l in self.kg_l.literal_set:
            if self.kg_r.literal_dict_by_value.__contains__(lite_l.value):
                lite_r = self.kg_r.literal_dict_by_value[lite_l.value]
                self.lite_align_dict[lite_l] = dict()
                self.lite_align_dict[lite_l][lite_r] = 1.0
                if self.lite_align_dict.__contains__(lite_r) is False:
                    self.lite_align_dict[lite_r] = dict()
                self.lite_align_dict[lite_r][lite_l] = 1.0
                self.lite_align_tuple_dict[(lite_l, lite_r)] = 1.0
                self.lite_align_tuple_dict[(lite_r, lite_l)] = 1.0

    def run(self, lite_matching=True, test_path=None):
        start_time = time.time()
        if lite_matching:
            new_ent_align_refined_dict, new_refined_tuple_dict = self.lite_align_dict.copy(), self.lite_align_tuple_dict.copy()
            new_ent_align_refined_dict.update(self.ent_align_refined_dict)
            new_refined_tuple_dict.update(self.refined_tuple_dict)
            self.ent_align_refined_dict = new_ent_align_refined_dict
            self.refined_tuple_dict = new_refined_tuple_dict
        print("Start...")
        for i in range(self.iteration):
            self._iter_num = i
            print(str(i + 1) + "-th iteration......")
            self.__run_per_iteration()
            if test_path is not None:
                print("Start testing...")
                for j in range(10):
                    self.util.test(path=test_path, threshold=0.1 * float(j))
        print("PARIS Completed!")
        end_time = time.time()
        print("Total time: " + str(end_time - start_time))

    def test(self, path, threshold=0.0):
        self.util.test(path=path, threshold=threshold)

    def save_results(self, path="output/EA_Result.txt"):
        self.util.save_results(path=path)

    def save_params(self, path="output/PARIS_Params"):
        self.util.save_params(path=path)

    def load_params(self, path="output/PARIS_Params"):
        self.util.load_params(path=path)

    def __run_per_iteration(self):
        print("Alignment...")
        self.__ent_rel_align_per_iteration()
        self.__norm_rel_attr_align_prob()
        self.__ent_bipartite_matching()
        self.__refine_rel_attr_candidate()
        self.__clear_candidate_dict()
        print("Complete an Iteration!")
        return

    def __ent_rel_align_per_iteration(self):
        kg_ent_list = list(self.kg_l.entity_set | self.kg_r.entity_set)
        random.shuffle(kg_ent_list)

        executor = ThreadPoolExecutor(max_workers=4)
        all_task = [executor.submit(self.__find_counterpart_of_ent, ent) for ent in kg_ent_list]
        wait(all_task, return_when=ALL_COMPLETED)

    def __find_counterpart_of_ent(self, ent):
        for (rel, ent_set) in ent.involved_as_tail_dict.items():
            if self._iter_num <= 2 and not rel.is_attribute():
                continue
            for head in ent_set:
                for (head_counterpart, head_eqv_prob) in self.__get_counterpart_dict(head).items():
                    if head_eqv_prob < self.theta:
                        continue
                    for (ent_counterpart, tail_eqv_prob) in self.__get_counterpart_dict(ent).items():
                        self.__register_rel_align_prob_norm(rel, head_eqv_prob * tail_eqv_prob)
                    for (rel_counterpart, head_counterpart_tail_set) in head_counterpart.involved_as_head_dict.items():
                        if rel.is_attribute() != rel_counterpart.is_attribute():
                            continue
                        for tail_counterpart in head_counterpart_tail_set:
                            tail_eqv_prob = max(self.__get_align_prob(ent, tail_counterpart), self.__get_align_prob(tail_counterpart, ent))
                            self.__register_ongoing_rel_align_prob(rel, rel_counterpart, 1.0 - head_eqv_prob * tail_eqv_prob)
                            self.__register_ent_equality(rel, ent, rel_counterpart, tail_counterpart, head_eqv_prob)
                        self.__update_rel_align_prob(rel, rel_counterpart)
        self.__update_ent_align_prob(ent)

    def __register_ent_equality(self, rel, tail, rel_counterpart, tail_counterpart, head_eqv_prob):
        prob_sub = self.__get_align_prob(rel, rel_counterpart)
        prob_sup = self.__get_align_prob(rel_counterpart, rel)
        if prob_sub < self.theta and prob_sup < self.theta:
            if self._iter_num <= 2:
                prob_sub, prob_sup = self.theta, self.theta
            else:
                return
        func_l, func_r = rel.functionality, rel_counterpart.functionality
        factor = 1.0
        factor_l = 1.0 - head_eqv_prob * prob_sup * func_r
        factor_r = 1.0 - head_eqv_prob * prob_sub * func_l
        if prob_sub >= 0.0 and func_l >= 0.0:
            factor *= factor_l
        if prob_sup >= 0.0 and func_r >= 0.0:
            factor *= factor_r
        if 1.0 - factor > self.epsilon:
            self.__register_ongoing_ent_align_prob(tail, tail_counterpart, factor)

    def __register_ongoing_ent_align_prob(self, ent_l, ent_r, prob):
        self.__register_ongoing_prob_product(self.ent_align_ongoing_dict, ent_l, ent_r, prob)
        return

    def __register_ongoing_rel_align_prob(self, rel_l, rel_r, prob):
        self.__register_ongoing_prob_product(self.rel_align_ongoing_dict, rel_l, rel_r, prob)
        return

    def __update_ent_align_prob(self, ent):
        counterpart, value = None, 0.0
        for (candidate, prob) in self.ent_align_ongoing_dict.get(ent, dict()).items():
            val = 1.0 - prob
            if val >= value:
                value, counterpart = val, candidate
        if value < self.theta or counterpart is None:
            return
        else:
            if self.ent_align_refined_dict.__contains__(ent) is False:
                self.ent_align_refined_dict[ent] = dict()
            self.ent_align_refined_dict[ent][counterpart] = value
            self.refined_tuple_dict[(ent, counterpart)] = value

    def __register_rel_align_prob_norm(self, rel, prob):
        if not self.rel_align_norm_dict.__contains__(rel):
            self.rel_align_norm_dict[rel] = 0.0
        self.rel_align_norm_dict[rel] += prob

    def __get_and_reset_ongoing_prob(self, rel_l, rel_r):
        if not self.rel_align_ongoing_dict.__contains__(rel_l):
            return 1.0
        if not self.rel_align_ongoing_dict[rel_l].__contains__(rel_r):
            return 1.0
        prob = self.rel_align_ongoing_dict[rel_l][rel_r]
        self.rel_align_ongoing_dict[rel_l][rel_r] = 1.0
        return prob

    def __update_rel_align_prob(self, rel_l, rel_r):
        if not self.rel_align_candidate_dict.__contains__(rel_l):
            self.rel_align_candidate_dict[rel_l] = dict()
        if not self.rel_align_candidate_dict[rel_l].__contains__(rel_r):
            self.rel_align_candidate_dict[rel_l][rel_r] = 0.0
        self.rel_align_candidate_dict[rel_l][rel_r] += 1.0 - self.__get_and_reset_ongoing_prob(rel_l, rel_r)

    def __get_counterpart_dict(self, obj):
        if obj.is_entity():
            return self.ent_align_refined_dict.get(obj, dict())
        else:
            return self.rel_align_refined_dict.get(obj, dict())

    def __norm_rel_attr_align_prob(self):
        new_rel_attr_align_dict = dict()
        for (obj, counterpart_dict) in self.rel_align_candidate_dict.items():
            norm = self.rel_align_norm_dict.get(obj, 1.0)
            for (counterpart, prob) in counterpart_dict.items():
                if norm == 0.0:
                    norm = 1.0
                norm_prob = prob / norm
                norm_prob = norm_prob if norm_prob <= 1.0 else 1.0
                norm_prob = norm_prob if norm_prob >= 0.0 else 0.0
                if new_rel_attr_align_dict.__contains__(obj) is False:
                    new_rel_attr_align_dict[obj] = dict()
                    new_rel_attr_align_dict[obj][counterpart] = norm_prob
                self.refined_tuple_candidate_dict[(obj, counterpart)] = norm_prob
        self.rel_align_candidate_dict = new_rel_attr_align_dict.copy()

    def __ent_bipartite_matching(self):
        visited, aligned_tuple_dict = set(), dict()

        def is_match(a, b):
            a_c, b_c = None, None
            for (e1, _) in self.ent_align_refined_dict[a].items():
                if e1 not in visited:
                    a_c = e1
                    break
            for (e2, _) in self.ent_align_refined_dict.get(b, dict()).items():
                if e2 not in visited:
                    b_c = e2
                    break
            return (a_c is b) and (b_c is a) and a_c is not None and b_c is not None

        def get_key(x):
            for (key, value) in x[1].items():
                return value

        for (ent, counterpart_dict) in self.ent_align_refined_dict.items():
            sorted(counterpart_dict.items(), key=lambda x: x[1], reverse=True)
        sorted(self.ent_align_refined_dict.items(), key=get_key, reverse=True)

        for (ent, counterpart_dict) in self.ent_align_refined_dict.items():
            for (counterpart, prob) in counterpart_dict.items():
                if counterpart in visited:
                    continue
                if is_match(ent, counterpart):
                    prob_inv = self.ent_align_refined_dict[counterpart][ent]
                    prob_assigned = max(prob, prob_inv)
                    aligned_tuple_dict[(ent, counterpart)] = prob_assigned
                    aligned_tuple_dict[(counterpart, ent)] = prob_assigned
                    visited.add(ent), visited.add(counterpart)
                    break
        if self.ent_candidate_num > 0:
            residual_ent_dict = dict()
            for (ent, counterpart_dict) in self.ent_align_refined_dict.items():
                if ent in visited:
                    continue
                for (counterpart, prob) in counterpart_dict.items():
                    if counterpart in visited:
                        continue
                    if not residual_ent_dict.__contains__(ent):
                        residual_ent_dict[ent] = dict()
                    residual_ent_dict[ent][counterpart] = prob

            for (ent, counterpart_dict) in residual_ent_dict.items():
                sorted(counterpart_dict.items(), key=lambda x: x[1], reverse=True)
            sorted(residual_ent_dict.items(), key=get_key, reverse=True)

            for (ent, counterpart_dict) in residual_ent_dict.items():
                if ent in visited:
                    continue
                candidate_num = 0
                for (counterpart, prob) in counterpart_dict.items():
                    if counterpart not in visited:
                        if candidate_num >= self.ent_candidate_num:
                            break
                        aligned_tuple_dict[(ent, counterpart)] = prob
                        aligned_tuple_dict[(counterpart, ent)] = prob
                        visited.add(ent), visited.add(counterpart)
                        candidate_num += 1

        self.ent_align_refined_dict.clear()
        self.__tuple_insert_helper(aligned_tuple_dict, self.ent_align_refined_dict)

        for ((l, r), p) in self.refined_tuple_dict.items():
            if l.is_relation():
                aligned_tuple_dict[(l, r)] = p
        self.refined_tuple_dict = aligned_tuple_dict

    @staticmethod
    def __tuple_insert_helper(tuple_dict: dict, target_dict: dict):
        for ((l, r), p) in tuple_dict.items():
            if not target_dict.__contains__(l):
                target_dict[l] = dict()
            target_dict[l][r] = p

    def __refine_rel_attr_candidate(self):
        rel_attr_align_dict, refined_tuple_dict = dict(), dict()

        for (obj_l, obj_r_dict) in self.rel_align_candidate_dict.items():
            candidate_dict = dict()
            sorted(obj_r_dict.items(), key=lambda x: x[1], reverse=True)
            candidate_num = 0
            for (candidate, prob) in obj_r_dict.items():
                candidate_dict[candidate] = prob
                refined_tuple_dict[(obj_l, candidate)] = prob
                candidate_num += 1
                if candidate_num >= self.rel_candidate_num:
                    break
            rel_attr_align_dict[obj_l] = candidate_dict.copy()

        for (obj_l, obj_r_dict) in self.ent_align_refined_dict.items():
            for (obj_r, prob) in obj_r_dict.items():
                refined_tuple_dict[(obj_l, obj_r)] = prob
        self.rel_align_refined_dict = rel_attr_align_dict.copy()
        self.refined_tuple_dict = refined_tuple_dict.copy()
        return

    def __clear_candidate_dict(self):
        self.rel_align_candidate_dict.clear()
        self.refined_tuple_candidate_dict.clear()
        self.rel_align_norm_dict.clear()
        self.ent_align_ongoing_dict.clear()
        self.rel_align_ongoing_dict.clear()

    def __get_align_prob(self, obj_l, obj_r):
        if self.refined_tuple_dict.__contains__((obj_l, obj_r)):
            return self.refined_tuple_dict[(obj_l, obj_r)]
        else:
            return 0.0

    @staticmethod
    def __register_ongoing_prob_product(dictionary, key1, key2, prob):
        if not dictionary.__contains__(key1):
            dictionary[key1] = dict()
        if not dictionary[key1].__contains__(key2):
            dictionary[key1][key2] = 1.0
        dictionary[key1][key2] *= prob
