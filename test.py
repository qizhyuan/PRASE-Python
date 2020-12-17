from objects.KG import KG
from objects.KGs import KGs
import os
import argparse


def construct_kg(path_r, path_a=None, sep='\t', name=None):
    kg = KG(name=name)
    if path_a is not None:
        with open(path_r, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = str.strip(line).split(sep=sep)
                if len(params) != 3:
                    print(line)
                    continue
                h, r, t = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_relation_tuple(h, r, t)

        with open(path_a, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = str.strip(line).split(sep=sep)
                if len(params) != 3:
                    print(line)
                    continue
                # assert len(params) == 3
                e, a, v = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_attribute_tuple(e, a, v)
    else:
        with open(path_r, "r", encoding="utf-8") as f:
            # pattern = "<([^>]+)>[ ]*<([^>]+)>[ ]*<?([^>]*)>?[ ]*\."
            # for line in f.readlines():
            #     matcher = re.match(pattern, line)
            #     e, a, v = matcher.group(1).strip(), matcher.group(2).strip(), matcher.group(3).strip()
            #     if len(e) == 0 or len(a) == 0 or len(v) == 0:
            #         print("Exception: " + e)
            #         continue
            #     if v.__contains__("http"):
            #         kg.insert_relation_tuple(e, a, v)
            #     else:
            #         kg.insert_attribute_tuple(e, a, v)
            prev_line = ""
            for line in f.readlines():
                params = line.strip().split(sep)
                if len(params) != 3 or len(prev_line) == 0:
                    prev_line += "\n" if len(line.strip()) == 0 else line.strip()
                    continue
                prev_params = prev_line.strip().split(sep)
                e, a, v = prev_params[0].strip(), prev_params[1].strip(), prev_params[2].strip()
                prev_line = "".join(line)
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
#
# path_r_2 = "dataset/person/person12.nt"

# path_r_1 = "dataset/ntriples/FMA_whole_ontology"
#
# path_r_2 = "dataset/ntriples/NCI_whole_ontology"

# path_r_1 = "dataset/EN_DE_100K_V2/rel_triples_1"
# path_a_1 = "dataset/EN_DE_100K_V2/attr_triples_1"

# path_r_2 = "dataset/EN_DE_100K_V2/rel_triples_2"
# path_a_2 = "dataset/EN_DE_100K_V2/attr_triples_2"

# path_validation = "dataset/EN_DE_100K_V2/ent_links"
# path_validation = "dataset/ntriples/FMA2NCI_mappings"
# kg1 = construct_kg(path_r_1, path_a_1, name="KG1")
# kg2 = construct_kg(path_r_2, path_a_2, name="KG2")

# kg1 = construct_kg(path_r=path_r_1, name="KG1", sep='\t')
# kg2 = construct_kg(path_r=path_r_2, name="KG2", sep='\t')

#
# kgs = KGs(kg1=kg1, kg2=kg2, iteration=20, theta=0.1, ent_candidate_num=0)
# kgs.run(test_path=path_validation)
# kgs.load_params()
# kgs.save_results("output/FMA2NCI/EA_Result.txt")
# kgs.save_params("output/FMA2NCI/EA_Params.txt")


def test(base, iteration=30):
    new_base, name = os.path.split(base)
    save_path = os.path.join(os.path.join("output", name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    path_r_1 = os.path.join(base, "rel_triples_1")
    path_a_1 = os.path.join(base, "attr_triples_1")

    path_r_2 = os.path.join(base, "rel_triples_2")
    path_a_2 = os.path.join(base, "attr_triples_2")

    path_validation = os.path.join(base, "ent_links")
    kg1 = construct_kg(path_r_1, path_a_1, name=str(name + "-KG1"))
    kg2 = construct_kg(path_r_2, path_a_2, name=str(name + "-KG2"))
    kgs = KGs(kg1=kg1, kg2=kg2, iteration=iteration, theta=0.1, ent_candidate_num=0)
    kgs.run(test_path=path_validation)
    kgs.save_results(os.path.join(save_path, "EA_Result.txt"))
    kgs.save_params(os.path.join(save_path, "EA_Params.txt"))


parser = argparse.ArgumentParser(description="PARIS_PYTHON")
parser.add_argument('--input', type=str)
parser.add_argument('--iteration', type=int, default=30)

args = parser.parse_args()

# test("dataset/industry", 3)

if __name__ == '__main__':
    test(args.input, args.iteration)

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

# kgs.validate(path_validation, threshold=0.0)
# kgs.store_params()

# for i in range(9):
#     validate_threshold = 0.1 * float(i)
#     kgs.validate(path_validation, validate_threshold)

# kgs.store_results()
# kgs.store_params()
