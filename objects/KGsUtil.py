import os


class KGsUtil:
    def __init__(self, kgs, get_counterpart_and_prob):
        self.kgs = kgs
        self.__get_counterpart_and_prob = get_counterpart_and_prob

    def test(self, path, threshold):
        correct_num, total_num = 0.0, 0.0
        ent_align_result = set()
        for ent_l in self.kgs.kg_l.entity_set:
            ent_id = ent_l.id
            counterpart = self.kgs.sub_ent_match[ent_id]
            if counterpart is not None:
                prob = self.kgs.sub_ent_prob[ent_id]
                if prob < threshold:
                    continue
                ent_align_result.add((ent_l, counterpart))

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
                if (obj_l, obj_r) in ent_align_result:
                    correct_num += 1.0
                total_num += 1.0

        if total_num == 0.0:
            print("Threshold: " + format(threshold, ".3f") + "\tException: no satisfied instance for testing")
        else:
            precision, recall = correct_num / len(ent_align_result), correct_num / total_num
            print("Threshold: " + format(threshold, ".3f") + "\tPrecision: " + format(precision, ".6f") +
                  "\tRecall: " + format(recall, ".6f"))

    def save_results(self, path):
        ent_dict, lite_dict, attr_dict, rel_dict = dict(), dict(), dict(), dict()
        for obj in (self.kgs.kg_l.entity_set | self.kgs.kg_l.literal_set):
            counterpart, prob = self.__get_counterpart_and_prob(obj)
            if counterpart is not None:
                if obj.is_literal():
                    lite_dict[(obj, counterpart)] = [prob]
                else:
                    ent_dict[(obj, counterpart)] = [prob]

        for (rel, rel_counterpart_dict) in self.kgs.rel_align_dict.items():
            if rel.affiliation is self.kgs.kg_r:
                continue
            dictionary = attr_dict if rel.is_attribute() else rel_dict
            for (rel_counterpart, prob) in rel_counterpart_dict.items():
                if prob > self.kgs.theta:
                    dictionary[(rel, rel_counterpart)] = [prob, 0.0]

        for (rel, rel_counterpart_dict) in self.kgs.rel_align_dict.items():
            if rel.affiliation is self.kgs.kg_l:
                continue
            dictionary = attr_dict if rel.is_attribute() else rel_dict
            for (rel_counterpart, prob) in rel_counterpart_dict.items():
                if prob > self.kgs.theta:
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
            for (rel, rel_counterpart_dict) in self.kgs.rel_align_dict.items():
                for (rel_counterpart, prob) in rel_counterpart_dict.items():
                    if prob > 0.0:
                        prefix = "L" if rel.affiliation is self.kgs.kg_l else "R"
                        f.write("\t".join([prefix, rel.name, rel_counterpart.name, str(prob)]) + "\n")
        return

    def load_params(self, path):
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
                        self.kgs.sub_ent_match[idx_l], self.kgs.sub_ent_prob[idx_l] = obj_r, prob
                    else:
                        self.kgs.sup_ent_match[idx_l], self.kgs.sup_ent_prob[idx_l] = obj_r, prob
                else:
                    self.__params_loader_helper(self.kgs.rel_align_dict, obj_l, obj_r, prob)
        return

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
