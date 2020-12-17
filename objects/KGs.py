import time
import gc
import random
from queue import Queue
from objects.KG import KG
from objects.KGsUtil import KGsUtil
from objects.Mapper import Mapper


class KGs:
    def __init__(self, kg1: KG, kg2: KG, ent_candidate_num=1, rel_candidate_num=1, theta=0.1, iteration=3):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = theta
        self.ent_candidate_num = ent_candidate_num
        self.rel_candidate_num = rel_candidate_num
        self.iteration = iteration
        self.delta = 0.01
        self.epsilon = 1.01
        self.const = 10.0

        self.rel_ongoing_dict = dict()
        self.rel_norm_dict = dict()
        self.rel_align_dict = dict()

        self.sub_ent_match = None
        self.sup_ent_match = None
        self.sub_ent_prob = None
        self.sup_ent_prob = None

        self._iter_num = 0
        self.util = KGsUtil(self, self.__get_counterpart_and_prob)
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
                self.sub_ent_match[l_id], self.sup_ent_match[r_id] = lite_r, lite_l
                self.sub_ent_prob[l_id], self.sup_ent_prob[r_id] = 1.0, 1.0

    def __get_counterpart_and_prob(self, ent):
        source = ent.affiliation is self.kg_l
        counterpart = self.sub_ent_match[ent.id] if source else self.sup_ent_match[ent.id]
        if counterpart is None:
            return None, 0.0
        else:
            return counterpart, self.sub_ent_prob[ent.id] if source else self.sup_ent_prob[ent.id]

    def __set_counterpart_and_prob(self, ent_l, ent_r, prob):
        source = ent_l.affiliation is self.kg_l
        l_id, r_id = ent_l.id, ent_r.id
        curr_prob = self.sub_ent_prob[l_id] if source else self.sup_ent_prob[l_id]
        if prob < curr_prob:
            return
        if source:
            self.sub_ent_match[l_id], self.sub_ent_prob[l_id] = ent_r, prob
        else:
            self.sup_ent_match[l_id], self.sup_ent_prob[l_id] = ent_r, prob

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
        # print("Start one way...")
        self.__run_per_iteration_one_way(self.kg_l)
        # print("Finish one way...")
        gc.collect()
        # print("Matching...")
        self.__ent_bipartite_matching()
        # print("Start one way...")
        self.__run_per_iteration_one_way(self.kg_r, ent_align=False)
        # print("Finish one way...")
        gc.collect()
        print("Complete an Iteration!")
        return

    def __run_per_iteration_one_way(self, kg: KG, ent_align=True):
        # print("Generate queue...")
        ent_queue = self.__generate_queue(kg)
        # print("Mapper running...")
        mapper = Mapper(queue=ent_queue, get_counterpart_and_prob=self.__get_counterpart_and_prob,
                        set_counterpart_and_prob=self.__set_counterpart_and_prob, rel_align_dict=self.rel_align_dict,
                        iter_num=self._iter_num, theta=self.theta, epsilon=self.epsilon, delta=self.delta,
                        ent_align=ent_align)
        mapper.run()
        # thread.join()
        self.rel_ongoing_dict.clear(), self.rel_norm_dict.clear()
        # print("Mapper-INV running...")
        rel_ongoing_dict, rel_norm_dict = mapper.get_rel_align_result()
        self.__merge_rel_ongoing_dict(self.rel_ongoing_dict, rel_ongoing_dict, rel_norm_dict)
        self.__merge_rel_norm_dict(self.rel_norm_dict, rel_norm_dict)
        # print("Relation alignment updating...")
        self.__update_rel_align_dict()

    @staticmethod
    def __generate_queue(kg: KG):
        ent_queue = Queue(maxsize=len(kg.entity_set))
        ent_list = list(kg.entity_set)
        random.shuffle(ent_list)
        for ent in ent_list:
            ent_queue.put(ent)
        return ent_queue

    @staticmethod
    def __merge_rel_ongoing_dict(rel_dict_l, rel_dict_r, norm_dict_r):
        for (rel, rel_counterpart_dict) in rel_dict_r.items():
            if not norm_dict_r.__contains__(rel):
                continue
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

    def __update_rel_align_dict(self):
        for (rel, counterpart_dict) in self.rel_ongoing_dict.items():
            norm = self.rel_norm_dict.get(rel, 1.0)
            if not self.rel_align_dict.__contains__(rel):
                self.rel_align_dict[rel] = dict()
            self.rel_align_dict[rel].clear()
            for (counterpart, score) in counterpart_dict.items():
                prob = score / (self.const + norm)
                self.rel_align_dict[rel][counterpart] = prob

    def __ent_bipartite_matching(self):
        for ent_l in self.kg_l.entity_set:
            ent_id = ent_l.id
            counterpart, prob = self.sub_ent_match[ent_id], self.sub_ent_prob[ent_id]
            if counterpart is None:
                continue
            counterpart_id = counterpart.id
            counterpart_prob = self.sup_ent_prob[counterpart_id]
            if counterpart_prob < prob:
                self.sup_ent_match[counterpart_id] = ent_l
                self.sup_ent_prob[counterpart_id] = prob
        for ent_l in self.kg_l.entity_set:
            ent_id = ent_l.id
            sub_counterpart = self.sub_ent_match[ent_id]
            if sub_counterpart is None:
                continue
            sup_counterpart = self.sup_ent_match[sub_counterpart.id]
            if sup_counterpart is None:
                continue
            if sup_counterpart.id != ent_id:
                self.sub_ent_match[ent_id], self.sub_ent_prob[ent_id] = None, 0.0
