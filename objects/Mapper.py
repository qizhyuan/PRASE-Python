# from multiprocessing import Process
from threading import Thread
from queue import Queue


class Mapper(Thread):
    def __init__(self, queue: Queue, get_counterpart_and_prob, set_counterpart_and_prob, rel_align_dict,
                 iter_num, theta, epsilon, delta, ent_align=True):
        super(Mapper, self).__init__()
        self.queue = queue
        self._theta = theta
        self._iter_num = iter_num
        self._epsilon = epsilon
        self._delta = delta
        self.rel_align_dict = rel_align_dict.copy()
        self._ent_align = ent_align

        self._get_counterpart_and_prob = get_counterpart_and_prob
        self._set_counterpart_and_prob = set_counterpart_and_prob

        self.rel_ongoing_dict = dict()
        self.rel_norm_dict = dict()

    def run(self):
        self.__ent_queue_handler()

    def __ent_queue_handler(self):
        while not self.queue.empty():
            ent = self.queue.get()
            self.__ent_align_handler(ent)

    def __ent_align_handler(self, ent):
        ent_align_ongoing_dict = dict()
        for (rel, ent_set) in ent.involved_as_tail_dict.items():
            if self._iter_num <= 1 and not rel.is_attribute():
                continue
            for head in ent_set:
                head_counterpart, head_eqv_prob = self._get_counterpart_and_prob(head)
                if head_counterpart is None or head_eqv_prob < self._theta:
                    continue
                ent_counterpart, tail_eqv_prob = self._get_counterpart_and_prob(ent)
                if ent_counterpart is not None:
                    self.__register_rel_align_prob_norm(self.rel_norm_dict, rel, head_eqv_prob * tail_eqv_prob)
                for (rel_counterpart, head_counterpart_tail_set) in head_counterpart.involved_as_head_dict.items():
                    if rel.is_attribute() != rel_counterpart.is_attribute():
                        continue
                    for tail_counterpart in head_counterpart_tail_set:
                        eqv_prob = tail_eqv_prob if tail_counterpart is ent_counterpart else 0.0
                        self.__register_ongoing_rel_align_prob(rel, rel_counterpart, head_eqv_prob * eqv_prob)
                        if self._ent_align:
                            self.__register_ent_equality(ent_align_ongoing_dict, rel, ent, rel_counterpart,
                                                         tail_counterpart, head_eqv_prob)
        if self._ent_align:
            self.__update_ent_align_prob(ent_align_ongoing_dict, ent)

    def __register_ent_equality(self, ent_align_ongoing_dict, rel, tail, rel_counterpart, tail_counterpart,
                                head_eqv_prob):
        prob_sub = self.__get_rel_align_prob(rel, rel_counterpart) / self._epsilon
        prob_sup = self.__get_rel_align_prob(rel_counterpart, rel) / self._epsilon
        if prob_sub < self._theta and prob_sup < self._theta:
            if self._iter_num <= 1:
                prob_sub, prob_sup = self._theta, self._theta
            else:
                return
        func_l, func_r = rel.functionality / self._epsilon, rel_counterpart.functionality / self._epsilon
        factor = 1.0
        factor_l = 1.0 - head_eqv_prob * prob_sup * func_r
        factor_r = 1.0 - head_eqv_prob * prob_sub * func_l
        if prob_sub >= 0.0 and func_l >= 0.0:
            factor *= factor_l
        if prob_sup >= 0.0 and func_r >= 0.0:
            factor *= factor_r
        if 1.0 - factor > self._delta and not tail.is_literal() and not tail_counterpart.is_literal():
            if not ent_align_ongoing_dict.__contains__(tail_counterpart):
                ent_align_ongoing_dict[tail_counterpart] = 1.0
            ent_align_ongoing_dict[tail_counterpart] *= factor

    def __register_ongoing_rel_align_prob(self, rel_l, rel_r, prob):
        self.__register_ongoing_prob_product(self.rel_ongoing_dict, rel_l, rel_r, prob)
        return

    def __get_rel_align_prob(self, rel_l, rel_r):
        if not self.rel_align_dict.__contains__(rel_l):
            return 0.0
        if not self.rel_align_dict[rel_l].__contains__(rel_r):
            return 0.0
        prob = self.rel_align_dict[rel_l][rel_r]
        prob = 1.0 if prob > 1.0 else prob
        prob = 0.0 if prob < 0.0 else prob
        return prob

    def __update_ent_align_prob(self, ent_align_ongoing_dict, ent):
        counterpart, value = None, 0.0
        for (candidate, prob) in ent_align_ongoing_dict.items():
            val = 1.0 - prob
            if val >= value:
                value, counterpart = val, candidate
        if counterpart is None:
            return
        else:
            self._set_counterpart_and_prob(ent, counterpart, value)

    def get_rel_align_result(self):
        return self.rel_ongoing_dict, self.rel_norm_dict

    @staticmethod
    def __register_ongoing_prob_product(dictionary, key1, key2, prob):
        if not dictionary.__contains__(key1):
            dictionary[key1] = dict()
        if not dictionary[key1].__contains__(key2):
            dictionary[key1][key2] = 0.0
        dictionary[key1][key2] += prob

    @staticmethod
    def __register_rel_align_prob_norm(dictionary, rel, prob):
        if not dictionary.__contains__(rel):
            dictionary[rel] = 0.0
        dictionary[rel] += prob
