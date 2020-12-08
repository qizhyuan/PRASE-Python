from objects.KG import KG
from objects.KGs import KGs
import re


def construct_kg(path_r, path_a=None, sep='\t', name=None):
    kg = KG(name=name)
    if path_a is not None:
        with open(path_r, "r", encoding="utf-8") as f:
            for line in f.readlines():
                params = str.strip(line).split(sep=sep)
                assert len(params) == 3
                h, r, t = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_relation_tuple(h, r, t)

        with open(path_a, "r", encoding="utf-8") as f:
            for line in f.readlines():
                params = str.strip(line).split(sep=sep)
                assert len(params) == 3
                e, a, v = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_attribute_tuple(e, a, v)
    else:
        with open(path_r, "r", encoding="utf-8") as f:
            pattern = "<([^>]+)>[ ]*<([^>]+)>[ ]*<?([^>]*)>?[ ]*\."
            for line in f.readlines():
                matcher = re.match(pattern, line)
                e, a, v = matcher.group(1).strip(), matcher.group(2).strip(), matcher.group(3).strip()
                if len(e) == 0 or len(a) == 0 or len(v) == 0:
                    print("Exception: " + e)
                    continue
                if v.__contains__("http"):
                    kg.insert_relation_tuple(e, a, v)
                else:
                    kg.insert_attribute_tuple(e, a, v)
    kg.calculate_functionality()
    kg.print_kg_info()
    return kg


# path_r_1 = "dataset/person/person11.nt"

# path_r_2 = "dataset/person/person12.nt"

path_r_1 = "dataset/industry/rel_triples_1"
path_a_1 = "dataset/industry/attr_triples_1"
#
path_r_2 = "dataset/industry/rel_triples_2"
path_a_2 = "dataset/industry/attr_triples_2"
#
path_validation = "dataset/industry/ent_links"

kg1 = construct_kg(path_r_1, path_a_1, name="KG1")
kg2 = construct_kg(path_r_2, path_a_2, name="KG2")

# kg1 = construct_kg(path_r=path_r_1, name="KG1", sep=' ')
# kg2 = construct_kg(path_r=path_r_2, name="KG2", sep=' ')

# for lite in kg1.literal_set:
#     print(lite.name + "\t" + lite.value)
#
# for lite in kg2.literal_set:
#     print(lite.name + "\t" + lite.value)

#
kgs = KGs(kg1=kg1, kg2=kg2, iteration=3, ent_lite_candidate_num=3, rel_attr_candidate_num=3, output_threshold=0.8, refine_threshold=0.1, theta=0.2)

kgs.run()
kgs.store_results()
# kgs.load_params()

# path_test = "dataset/industry/ent_links"
# with open(path_test, "r", encoding="utf-8") as f:
#     num = 2000
#     for line in f.readlines():
#         params = str.strip(line).split(sep='\t')
#         assert len(params) == 2
#         e, a = params[0].strip(), params[1].strip()
#         kgs.insert_ent_tuple(e, a, 1.0)
#         num -= 1
#         if num <= 0:
#             break
# kgs.load_params()
# kgs.run()

kgs.validate(path_validation, threshold=0.2)
# kgs.store_params()

# for i in range(9):
#     validate_threshold = 0.1 * float(i)
#     kgs.validate(path_validation, validate_threshold)

# kgs.store_results()
# kgs.store_params()
