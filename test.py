from objects.KG import KG
from objects.KGs import KGs
import os
import argparse
import numpy as np


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

def fusion_func(prob, x, y):
    return 0.5 * prob + 0.5 * np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


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
    kgs = KGs(kg1=kg1, kg2=kg2, iteration=iteration, theta=0.1, workers=6)
    # kgs.run(test_path=path_validation)
    kgs.util.load_params(os.path.join(save_path, "EA_Params.txt"))
    # kgs.util.generate_input_for_embed_align(link_path=path_validation, save_dir=save_path, threshold=0.9)
    # bootea_links_path = os.path.join(save_path, "BootEA_EA_Result")
    # bootea_links_path = os.path.join(save_path, "BootEA_GT_Result")
    mtranse_links_path = os.path.join(save_path, "MTransE_EA_Result")
    # imuse_links_path = os.path.join(save_path, "IMUSE_EA_Result")
    # ent_links_path = os.path.join(save_path, "valid_links")
    kgs.util.reset_ent_align_prob(lambda x: 0.9 * x)
    ent_emb_path = os.path.join(save_path, "ent_embeds.npy")
    mapping_l, mapping_r = os.path.join(save_path, "kg1_ent_ids"), os.path.join(save_path, "kg2_ent_ids")
    kgs.util.load_embedding(ent_emb_path, mapping_l, mapping_r)

    kgs.set_fusion_func(fusion_func)
    kgs.util.load_ent_links(path=mtranse_links_path, func=lambda x: x / 2.0, threshold_min=0.2, threshold_max=1.0, force=True)
    kgs.util.test(path_validation, 0.1)
    kgs.run(test_path=path_validation)
    # kgs.util.save_results(os.path.join(save_path, "EA_Result.txt"))
    # kgs.util.save_params(os.path.join(save_path, "EA_Params.txt"))


parser = argparse.ArgumentParser(description="PARIS_PYTHON")
parser.add_argument('--input', type=str)
parser.add_argument('--iteration', type=int, default=30)

args = parser.parse_args()

if __name__ == '__main__':
    # test(args.input, args.iteration)
    test("dataset/D_W_100K_V2", 10)
    # test("dataset/industry", 10)
