import gc
import sys
import os
import time
import random
import numpy as np
from objects.KG import KG
from model.PARIS import one_iteration_one_way
import multiprocessing as mp

sys.setrecursionlimit(1000000)


class KGs:
    def __init__(self, kg1: KG, kg2: KG, theta=0.1, iteration=3, workers=4):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = theta
        self.iteration = iteration
        self.delta = 0.01
        self.epsilon = 1.01
        self.const = 10.0
        self.workers = workers

        self.rel_ongoing_dict_l, self.rel_ongoing_dict_r = dict(), dict()
        self.rel_norm_dict_l, self.rel_norm_dict_r = dict(), dict()
        self.rel_align_dict_l, self.rel_align_dict_r = dict(), dict()

        self.sub_ent_match = None
        self.sup_ent_match = None
        self.sub_ent_prob = None
        self.sup_ent_prob = None

        self._iter_num = 0
        self.has_load = False
        self.util = KGsUtil(self, self.__get_counterpart_and_prob, self.__set_counterpart_and_prob)
        self.__init()

    def __init(self):
        kg_l_ent_num = len(self.kg_l.entity_set) + len(self.kg_l.literal_set)
        kg_r_ent_num = len(self.kg_r.entity_set) + len(self.kg_r.literal_set)
        self.sub_ent_match = [None for _ in range(kg_l_ent_num)]
        self.sub_ent_prob = [0.0 for _ in range(kg_l_ent_num)]
        self.sup_ent_match = [None for _ in range(kg_r_ent_num)]
        self.sup_ent_prob = [0.0 for _ in range(kg_r_ent_num)]

        for lite_l in self.kg_l.literal_set:
            if self.kg_r.literal_dict_by_value.__contains__(lite_l.value):
                lite_r = self.kg_r.literal_dict_by_value[lite_l.value]
                l_id, r_id = lite_l.id, lite_r.id
                self.sub_ent_match[l_id], self.sup_ent_match[r_id] = lite_r.id, lite_l.id
                self.sub_ent_prob[l_id], self.sup_ent_prob[r_id] = 1.0, 1.0

    def __get_counterpart_and_prob(self, ent):
        source = ent.affiliation is self.kg_l
        counterpart_id = self.sub_ent_match[ent.id] if source else self.sup_ent_match[ent.id]
        if counterpart_id is None:
            return None, 0.0
        else:
            counterpart = self.kg_r.entity_dict_by_id.get(counterpart_id) if source \
                else self.kg_l.entity_dict_by_id.get(counterpart_id)
            return counterpart, self.sub_ent_prob[ent.id] if source else self.sup_ent_prob[ent.id]

    def __set_counterpart_and_prob(self, ent_l, ent_r, prob):
        source = ent_l.affiliation is self.kg_l
        l_id, r_id = ent_l.id, ent_r.id
        curr_prob = self.sub_ent_prob[l_id] if source else self.sup_ent_prob[l_id]
        if prob < curr_prob:
            return
        if source:
            self.sub_ent_match[l_id], self.sub_ent_prob[l_id] = r_id, prob
        else:
            self.sup_ent_match[l_id], self.sup_ent_prob[l_id] = r_id, prob

    def run(self, test_path=None):
        start_time = time.time()
        print("Start...")
        for i in range(self.iteration):
            self._iter_num = i
            print(str(i + 1) + "-th iteration......")
            self.__run_per_iteration()
            if test_path is not None:
                print("Start testing...")
                for j in range(10):
                    self.util.test(path=test_path, threshold=0.1 * float(j))
            gc.collect()
            # self.reset_ent_align_prob(lambda x: 0.95 * x)
            # if i % 2 == 0:
            #     self.util.set_random_candidate()
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

    def load_ea_result(self, path, init_value=0.3):
        self.util.load_ea_result(path, init_value)

    def load_ent_links(self, path, init_value=0.3, num=100, threshold=0.9):
        self.util.load_ent_links(path=path, init_value=init_value, num=num, threshold=threshold)

    def load_multi_ent_links(self, init_value=0.3, *paths):
        self.util.load_multi_ent_links(init_value, *paths)

    def reset_ent_align_prob(self, func):
        for ent in self.kg_l.entity_set:
            idx = ent.id
            self.sub_ent_prob[idx] = func(self.sub_ent_prob[idx])
        for ent in self.kg_r.entity_set:
            idx = ent.id
            self.sup_ent_prob[idx] = func(self.sup_ent_prob[idx])

    # def save_seeds(self, path="output/PARIS_Seeds", threshold=0.9):
    #     self.util.save_seeds(path=path, threshold=threshold)

    def generate_new_dataset(self, link_path, save_dir, threshold=0.1):
        self.util.generate_new_dataset(link_path, save_dir, threshold)

    def __run_per_iteration(self):
        self.__run_per_iteration_one_way(self.kg_l)
        self.__ent_bipartite_matching()
        self.__run_per_iteration_one_way(self.kg_r, ent_align=False)
        return

    def __run_per_iteration_one_way(self, kg: KG, ent_align=True):
        kg_other = self.kg_l if kg is self.kg_r else self.kg_r
        ent_list = self.__generate_list(kg)
        mgr = mp.Manager()
        ent_queue = mgr.Queue(len(ent_list))
        for ent_id in ent_list:
            ent_queue.put(ent_id)

        rel_ongoing_dict_queue = mgr.Queue()
        rel_norm_dict_queue = mgr.Queue()
        ent_match_tuple_queue = mgr.Queue()

        kg_r_fact_dict_by_head = kg_other.fact_dict_by_head
        kg_l_fact_dict_by_tail = kg.fact_dict_by_tail
        kg_l_func, kg_r_func = kg.functionality_dict, kg_other.functionality_dict

        rel_align_dict_l, rel_align_dict_r = self.rel_align_dict_l, self.rel_align_dict_r

        if kg is self.kg_l:
            ent_match, ent_prob = self.sub_ent_match, self.sub_ent_prob
            is_literal_list_r = self.kg_r.is_literal_list
        else:
            ent_match, ent_prob = self.sup_ent_match, self.sup_ent_prob
            rel_align_dict_l, rel_align_dict_r = rel_align_dict_r, rel_align_dict_l
            is_literal_list_r = self.kg_l.is_literal_list

        init = not self.has_load and self._iter_num <= 1
        tasks = []
        kg_l_ent_embeds, kg_r_ent_embeds = kg.ent_embeddings, kg_other.ent_embeddings
        for _ in range(self.workers):
            task = mp.Process(target=one_iteration_one_way, args=(ent_queue, kg_r_fact_dict_by_head,
                                                                  kg_l_fact_dict_by_tail,
                                                                  kg_l_func, kg_r_func,
                                                                  ent_match, ent_prob,
                                                                  is_literal_list_r,
                                                                  rel_align_dict_l, rel_align_dict_r,
                                                                  rel_ongoing_dict_queue, rel_norm_dict_queue,
                                                                  ent_match_tuple_queue,
                                                                  kg_l_ent_embeds, kg_r_ent_embeds,
                                                                  self.theta, self.epsilon, self.delta, init,
                                                                  ent_align))
            task.start()
            tasks.append(task)

        for task in tasks:
            task.join()

        self.__clear_ent_match_and_prob(ent_match, ent_prob)
        while not ent_match_tuple_queue.empty():
            ent_match_tuple = ent_match_tuple_queue.get()
            self.__merge_ent_align_result(ent_match, ent_prob, ent_match_tuple[0], ent_match_tuple[1])

        rel_ongoing_dict = self.rel_ongoing_dict_l if kg is self.kg_l else self.rel_ongoing_dict_r
        rel_norm_dict = self.rel_norm_dict_l if kg is self.kg_l else self.rel_norm_dict_r
        rel_align_dict = self.rel_align_dict_l if kg is self.kg_l else self.rel_align_dict_r

        rel_ongoing_dict.clear(), rel_norm_dict.clear(), rel_align_dict.clear()
        while not rel_ongoing_dict_queue.empty():
            self.__merge_rel_ongoing_dict(rel_ongoing_dict, rel_ongoing_dict_queue.get())

        while not rel_norm_dict_queue.empty():
            self.__merge_rel_norm_dict(rel_norm_dict, rel_norm_dict_queue.get())

        self.__update_rel_align_dict(rel_align_dict, rel_ongoing_dict, rel_norm_dict)

    @staticmethod
    def __generate_list(kg: KG):
        ent_list = kg.ent_id_list
        random.shuffle(ent_list)
        return ent_list

    @staticmethod
    def __merge_rel_ongoing_dict(rel_dict_l, rel_dict_r):
        for (rel, rel_counterpart_dict) in rel_dict_r.items():
            if not rel_dict_l.__contains__(rel):
                rel_dict_l[rel] = rel_counterpart_dict
            else:
                for (rel_counterpart, prob) in rel_counterpart_dict.items():
                    if not rel_dict_l[rel].__contains__(rel_counterpart):
                        rel_dict_l[rel][rel_counterpart] = prob
                    else:
                        rel_dict_l[rel][rel_counterpart] += prob

    @staticmethod
    def __merge_rel_norm_dict(norm_dict_l, norm_dict_r):
        for (rel, norm) in norm_dict_r.items():
            if not norm_dict_l.__contains__(rel):
                norm_dict_l[rel] = norm
            else:
                norm_dict_l[rel] += norm

    @staticmethod
    def __update_rel_align_dict(rel_align_dict, rel_ongoing_dict, rel_norm_dict, const=10.0):
        for (rel, counterpart_dict) in rel_ongoing_dict.items():
            norm = rel_norm_dict.get(rel, 1.0)
            if not rel_align_dict.__contains__(rel):
                rel_align_dict[rel] = dict()
            rel_align_dict[rel].clear()
            for (counterpart, score) in counterpart_dict.items():
                prob = score / (const + norm)
                rel_align_dict[rel][counterpart] = prob

    def __ent_bipartite_matching(self):
        for ent_l in self.kg_l.entity_set:
            ent_id = ent_l.id
            counterpart_id, prob = self.sub_ent_match[ent_id], self.sub_ent_prob[ent_id]
            if counterpart_id is None:
                continue
            counterpart_prob = self.sup_ent_prob[counterpart_id]
            if counterpart_prob < prob:
                self.sup_ent_match[counterpart_id] = ent_id
                self.sup_ent_prob[counterpart_id] = prob
        for ent_l in self.kg_l.entity_set:
            ent_id = ent_l.id
            sub_counterpart_id = self.sub_ent_match[ent_id]
            if sub_counterpart_id is None:
                continue
            sup_counterpart_id = self.sup_ent_match[sub_counterpart_id]
            if sup_counterpart_id is None:
                continue
            if sup_counterpart_id != ent_id:
                self.sub_ent_match[ent_id], self.sub_ent_prob[ent_id] = None, 0.0

    @staticmethod
    def __merge_ent_align_result(ent_match_l, ent_prob_l, ent_match_r, ent_prob_r):
        assert len(ent_match_l) == len(ent_match_r)
        for i in range(len(ent_prob_l)):
            if ent_prob_l[i] < ent_prob_r[i]:
                ent_prob_l[i] = ent_prob_r[i]
                ent_match_l[i] = ent_match_r[i]

    @staticmethod
    def __clear_ent_match_and_prob(ent_match, ent_prob):
        for i in range(len(ent_match)):
            ent_match[i] = None
            ent_prob[i] = 0.0


class KGsUtil:
    def __init__(self, kgs, get_counterpart_and_prob, set_counterpart_and_prob):
        self.kgs = kgs
        self.__get_counterpart_and_prob = get_counterpart_and_prob
        self.__set_counterpart_and_prob = set_counterpart_and_prob
        self.ent_links_candidate = list()

    def test(self, path, threshold):
        correct_num, total_num = 0.0, 0.0
        ent_align_result = set()
        for ent_id in self.kgs.kg_l.ent_id_list:
            counterpart_id = self.kgs.sub_ent_match[ent_id]
            if counterpart_id is not None:
                prob = self.kgs.sub_ent_prob[ent_id]
                if prob < threshold:
                    continue
                ent_align_result.add((ent_id, counterpart_id))

        if len(ent_align_result) == 0:
            print("Threshold: " + format(threshold, ".3f") + "\tException: no satisfied alignment result")
            return
        with open(path, "r", encoding="utf8") as f:
            for line in f.readlines():
                params = str.strip(line).split("\t")
                ent_l, ent_r = params[0].strip(), params[1].strip()
                obj_l, obj_r = self.kgs.kg_l.entity_dict_by_name.get(ent_l), self.kgs.kg_r.entity_dict_by_name.get(ent_r)
                if obj_l is None:
                    print("Exception: fail to load Entity (" + ent_l + ")")
                if obj_r is None:
                    print("Exception: fail to load Entity (" + ent_r + ")")
                if obj_l is None or obj_r is None:
                    continue
                if (obj_l.id, obj_r.id) in ent_align_result:
                    correct_num += 1.0
                total_num += 1.0

        if total_num == 0.0:
            print("Threshold: " + format(threshold, ".3f") + "\tException: no satisfied instance for testing")
        else:
            precision, recall = correct_num / len(ent_align_result), correct_num / total_num
            print("Threshold: " + format(threshold, ".3f") + "\tPrecision: " + format(precision, ".6f") +
                  "\tRecall: " + format(recall, ".6f"))

    def generate_new_dataset(self, link_path, save_dir, threshold):
        ent_align_predict = set()
        for ent in self.kgs.kg_l.entity_set:
            counterpart, prob = self.__get_counterpart_and_prob(ent)
            if prob < threshold:
                continue
            ent_align_predict.add((ent, counterpart))

        ent_align_test = set()
        with open(link_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                params = str.strip(line).split("\t")
                ent_l, ent_r = params[0].strip(), params[1].strip()
                obj_l, obj_r = self.kgs.kg_l.entity_dict_by_name.get(ent_l), self.kgs.kg_r.entity_dict_by_name.get(ent_r)
                if obj_l is None or obj_r is None:
                    continue
                if (obj_l, obj_r) not in ent_align_predict:
                    ent_align_test.add((obj_l, obj_r))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_path = os.path.join(save_dir, "train_links")
        test_path = os.path.join(save_dir, "test_links")
        valid_path = os.path.join(save_dir, "valid_links")

        def writer(save_path, result_set):
            with open(save_path, "w", encoding="utf8") as f:
                num, length = 0, len(result_set)
                for (obj_l, obj_r) in result_set:
                    f.write("\t".join([obj_l.name, obj_r.name]))
                    num += 1
                    if num < length:
                        f.write("\n")

        writer(train_path, ent_align_predict)
        writer(test_path, ent_align_test)
        writer(valid_path, ent_align_test)

    def save_results(self, path):
        ent_dict, lite_dict, attr_dict, rel_dict = dict(), dict(), dict(), dict()
        for obj in (self.kgs.kg_l.entity_set | self.kgs.kg_l.literal_set):
            counterpart, prob = self.__get_counterpart_and_prob(obj)
            if counterpart is not None:
                if obj.is_literal():
                    lite_dict[(obj, counterpart)] = [prob]
                else:
                    ent_dict[(obj, counterpart)] = [prob]

        for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_l.items():
            rel = self.kgs.kg_l.relation_dict_by_id.get(rel_id)
            dictionary = attr_dict if rel.is_attribute() else rel_dict
            for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                if prob > self.kgs.theta:
                    rel_counterpart = self.kgs.kg_r.relation_dict_by_id.get(rel_counterpart_id)
                    dictionary[(rel, rel_counterpart)] = [prob, 0.0]

        for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_r.items():
            rel = self.kgs.kg_r.relation_dict_by_id.get(rel_id)
            dictionary = attr_dict if rel.is_attribute() else rel_dict
            for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                if prob > self.kgs.theta:
                    rel_counterpart = self.kgs.kg_l.relation_dict_by_id.get(rel_counterpart_id)
                    if not dictionary.__contains__((rel_counterpart, rel)):
                        dictionary[(rel_counterpart, rel)] = [0.0, 0.0]
                    dictionary[(rel_counterpart, rel)][-1] = prob
        base, _ = os.path.split(path)
        if not os.path.exists(base):
            os.makedirs(base)
        if os.path.exists(path):
            os.remove(path)
        self.__result_writer(path, attr_dict, "Attribute Alignment")
        self.__result_writer(path, rel_dict, "Relation Alignment")
        self.__result_writer(path, lite_dict, "Literal Alignment")
        self.__result_writer(path, ent_dict, "Entity Alignment")
        return

    def save_params(self, path):
        base, _ = os.path.split(path)
        if not os.path.exists(base):
            os.makedirs(base)
        with open(path, "w", encoding="utf-8") as f:
            for obj in (self.kgs.kg_l.entity_set | self.kgs.kg_l.literal_set):
                counterpart, prob = self.__get_counterpart_and_prob(obj)
                if counterpart is not None:
                    f.write("\t".join(["L", obj.name, counterpart.name, str(prob)]) + "\n")
            for obj in (self.kgs.kg_r.entity_set | self.kgs.kg_r.literal_set):
                counterpart, prob = self.__get_counterpart_and_prob(obj)
                if counterpart is not None:
                    f.write("\t".join(["R", obj.name, counterpart.name, str(prob)]) + "\n")
            for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_l.items():
                rel = self.kgs.kg_l.relation_dict_by_id.get(rel_id)
                for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                    if prob > 0.0:
                        rel_counterpart = self.kgs.kg_r.relation_dict_by_id.get(rel_counterpart_id)
                        prefix = "L"
                        f.write("\t".join([prefix, rel.name, rel_counterpart.name, str(prob)]) + "\n")
            for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_r.items():
                rel = self.kgs.kg_r.relation_dict_by_id.get(rel_id)
                for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                    if prob > 0.0:
                        rel_counterpart = self.kgs.kg_l.relation_dict_by_id.get(rel_counterpart_id)
                        prefix = "R"
                        f.write("\t".join([prefix, rel.name, rel_counterpart.name, str(prob)]) + "\n")
        return

    def load_params(self, path):
        self.kgs.has_load = True
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = line.strip().split("\t")
                assert len(params) == 4
                prefix, name_l, name_r, prob = params[0].strip(), params[1].strip(), params[2].strip(), float(params[3].strip())
                if prefix == "L":
                    obj_l, obj_r = self.kgs.kg_l.get_object_by_name(name_l), self.kgs.kg_r.get_object_by_name(name_r)
                else:
                    obj_l, obj_r = self.kgs.kg_r.get_object_by_name(name_l), self.kgs.kg_l.get_object_by_name(name_r)
                assert (obj_l is not None and obj_r is not None)
                if obj_l.is_entity():
                    idx_l = obj_l.id
                    if prefix == "L":
                        self.kgs.sub_ent_match[idx_l], self.kgs.sub_ent_prob[idx_l] = obj_r.id, prob
                    else:
                        self.kgs.sup_ent_match[idx_l], self.kgs.sup_ent_prob[idx_l] = obj_r.id, prob
                else:
                    if prefix == "L":
                        self.__params_loader_helper(self.kgs.rel_align_dict_l, obj_l.id, obj_r.id, prob)
                    else:
                        self.__params_loader_helper(self.kgs.rel_align_dict_r, obj_l.id, obj_r.id, prob)
        return

    def load_ent_links(self, path, init_value=0.3, num=100, threshold=0.8, epsilon=5):
        idx = 0
        with open(path, "r", encoding="utf8") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                params = line.split(sep="\t")
                if len(params) == 3:
                    name_l, name_r, prob = params[0].strip(), params[1].strip(), float(params[2].strip())
                    if prob < 0.8 or prob > 0.95:
                        continue
                    obj_l, obj_r = self.kgs.kg_l.get_object_by_name(name_l), self.kgs.kg_r.get_object_by_name(name_r)
                    if obj_l is None or obj_r is None:
                        continue
                    self.__set_counterpart_and_prob(obj_l, obj_r, prob / epsilon)
                    # self.__set_counterpart_and_prob(obj_r, obj_l, prob / epsilon)
                elif len(params) == 2:
                    name_l, name_r = params[0].strip(), params[1].strip()
                    obj_l, obj_r = self.kgs.kg_l.get_object_by_name(name_l), self.kgs.kg_r.get_object_by_name(name_r)
                    if obj_l is None or obj_r is None:
                        continue
                    self.__set_counterpart_and_prob(obj_l, obj_r, 1.0 / epsilon)
                    self.__set_counterpart_and_prob(obj_r, obj_l, 1.0 / epsilon)
                else:
                    continue
                idx += 1
                if idx >= num:
                    break
        print("load num: " + str(idx))

    def load_multi_ent_links(self, init_value, *paths):
        ent_links_set = set()

        def get_ent_links_set(path):
            ent_link_set_tmp = set()
            with open(path, "r", encoding="utf8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    params = line.split(sep="\t")
                    name_l, name_r, prob = params[0].strip(), params[1].strip(), float(params[2].strip())
                    if prob < 0.6:
                        continue
                    obj_l, obj_r = self.kgs.kg_l.get_object_by_name(name_l), self.kgs.kg_r.get_object_by_name(name_r)
                    if obj_l is None or obj_r is None:
                        continue
                    ent_link_set_tmp.add((obj_l, obj_r))
            return ent_link_set_tmp

        ent_links_set_list = []
        for path in paths:
            ent_links_set_list.append(get_ent_links_set(path))
            ent_links_set |= ent_links_set_list[-1]
        for ent_links_set_item in ent_links_set_list:
            ent_links_set = ent_links_set & ent_links_set_item

        print(len(ent_links_set))
        for (obj_l, obj_r) in ent_links_set:
            self.ent_links_candidate.append((obj_l, obj_r))
            # self.__set_counterpart_and_prob(obj_l, obj_r, init_value)
            # self.__set_counterpart_and_prob(obj_r, obj_l, init_value)

    def set_random_candidate(self, num=500, init_value=0.2):
        random_list = random.choices(self.ent_links_candidate, k=num)
        for (obj_l, obj_r) in random_list:
            self.__set_counterpart_and_prob(obj_l, obj_r, init_value)
            self.__set_counterpart_and_prob(obj_r, obj_l, init_value)

    def load_ea_result(self, path, init_value=0.3):
        align_path = os.path.join(path, "ea_alignment_results")
        kg1_ent_idx_path = os.path.join(path, "kg1_ent_ids")
        kg2_ent_idx_path = os.path.join(path, "kg2_ent_ids")
        kg_idx_dict = dict()

        def generate_idx_dict(path, dictionary):
            with open(path, "r", encoding="utf8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    params = line.split(sep="\t")
                    name, idx = params[0].strip(), int(params[1].strip())
                    dictionary[idx] = name

        generate_idx_dict(kg1_ent_idx_path, kg_idx_dict)
        generate_idx_dict(kg2_ent_idx_path, kg_idx_dict)

        with open(align_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                params = line.split(sep="\t")
                idx_l, idx_r = 2 * int(params[0].strip()), 2 * int(params[1].strip()) + 1
                name_l, name_r = kg_idx_dict.get(idx_l, None), kg_idx_dict.get(idx_r, None)
                obj_l, obj_r = self.kgs.kg_l.get_object_by_name(name_l), self.kgs.kg_r.get_object_by_name(name_r)
                if obj_l is None or obj_r is None:
                    continue
                self.__set_counterpart_and_prob(obj_l, obj_r, init_value)
                self.__set_counterpart_and_prob(obj_r, obj_l, init_value)

    def load_embedding(self, ent_emb_path, kg_l_mapping, kg_r_mapping):
        ent_emb = np.load(ent_emb_path)

        def load_emb_helper(kg, mapping_path):
            with open(mapping_path, "r", encoding="utf8") as f:
                for line in f.readlines():
                    if len(line.strip()) == 0:
                        continue
                    params = line.strip().split("\t")
                    ent_name, idx = params[0].strip(), int(params[1].strip())
                    ent = kg.entity_dict_by_name.get(ent_name)
                    if ent is not None:
                        ent.embedding = ent_emb[idx, :]

        load_emb_helper(self.kgs.kg_l, kg_l_mapping)
        load_emb_helper(self.kgs.kg_r, kg_r_mapping)
        self.kgs.kg_l.init_ent_embeddings()
        self.kgs.kg_r.init_ent_embeddings()

    @staticmethod
    def __result_writer(path, result_dict, title):
        with open(path, "a+", encoding="utf-8") as f:
            f.write("--- " + title + " ---\n\n")
            for ((obj_l, obj_r), prob_set) in result_dict.items():
                f.write(obj_l.name + "\t" + obj_r.name + "\t" + "\t".join(format(s, ".6f") for s in prob_set) + "\n")
            f.write("\n")

    @staticmethod
    def __params_loader_helper(dict_by_key: dict, key1, key2, value):
        if not dict_by_key.__contains__(key1):
            dict_by_key[key1] = dict()
        dict_by_key[key1][key2] = value
