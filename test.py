import argparse
import os

import numpy as np

from objects.KG import KG
from objects.KGs import KGs


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
    kg.init()
    kg.print_kg_info()
    return kg


def fusion_func_8_2(prob, x, y):
    return 0.8 * prob + 0.2 * np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def fusion_func_5_5(prob, x, y):
    return 0.5 * prob + 0.5 * np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def test(base, func=None, emb_name=None, iteration=10, worker=6, load_weight=1.0, reset_weight=1.0, load_ent=False, load_emb=False, init_reset=False):
    new_base, name = os.path.split(base)
    save_path = os.path.join(os.path.join("output", name))
    print("\t".join(["path:", base, "emb_name:", emb_name, "iteration:", str(iteration), "worker:", str(worker), "load_weight:", str(load_weight), "reset_weight:",
                     str(reset_weight), "load_ent:", str(load_ent), "load_emb:", str(load_emb), "init_reset:", str(init_reset)]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if func is fusion_func_8_2:
        print("func: fusion_func_8_2")
    if func is fusion_func_5_5:
        print("func: fusion_func_5_5")

    path_r_1 = os.path.join(base, "rel_triples_1")
    path_a_1 = os.path.join(base, "attr_triples_1")

    path_r_2 = os.path.join(base, "rel_triples_2")
    path_a_2 = os.path.join(base, "attr_triples_2")

    path_validation = os.path.join(base, "ent_links")
    kg1 = construct_kg(path_r_1, path_a_1, name=str(name + "-KG1"))
    kg2 = construct_kg(path_r_2, path_a_2, name=str(name + "-KG2"))
    kgs = KGs(kg1=kg1, kg2=kg2, iteration=iteration, theta=0.1, workers=worker)

    kgs.util.test(path_validation, [0.0, 0.1])

    if init_reset is True:
        kgs.util.reset_ent_align_prob(lambda x: reset_weight * x)

    if load_ent is True:
        base_path = os.path.join(save_path, emb_name)
        ent_links_path = os.path.join(base_path, "alignment_results_12")
        kgs.util.load_ent_links(func=lambda x: load_weight * x, path=ent_links_path, force=True)

    if load_emb is True:
        base_path = os.path.join(save_path, emb_name)
        mapping_l, mapping_r = os.path.join(base_path, "kg1_ent_ids"), os.path.join(base_path, "kg2_ent_ids")
        ent_emb_path = os.path.join(base_path, "ent_embeds.npy")
        kgs.util.load_embedding(ent_emb_path, mapping_l, mapping_r)
        print("load embedding...")
        kgs.set_fusion_func(func)

    kgs.run(test_path=path_validation)
    # save_path = os.path.join(save_path, "links")
    # kgs.util.generate_input_for_embed_align(link_path=path_validation, save_dir=save_path, threshold=0.1)
    # kgs.util.save_results(os.path.join(save_path, "EA_Result.txt"))
    # kgs.util.save_params(os.path.join(save_path, "EA_Params.txt"))


parser = argparse.ArgumentParser(description="PARIS_PYTHON")

parser.add_argument('--model', type=str, default="PARIS")
parser.add_argument('--base', type=str)
parser.add_argument('--worker', type=int, default=6)

args = parser.parse_args()

if __name__ == '__main__':
    name, base, worker = args.model, args.base, args.worker
    data_list = ["industry", "D_W_15K_V2", "D_W_100K_V2", "EN_DE_100K_V2", "EN_FR_100K_V2", "D_Y_100K_V2"]
    for data_name in data_list:
        path = os.path.join(base, data_name)
        # test(base=path, emb_name=name, iteration=10, worker=worker, load_weight=1.0, reset_weight=1.0, load_ent=False,
        #      init_reset=False)
        # test(base=path, emb_name=name, iteration=10, worker=worker, load_weight=1.0, reset_weight=1.0, load_ent=True,
        #      init_reset=False)
        # test(base=path, emb_name=name, iteration=10, worker=worker, load_weight=0.5, reset_weight=1.0, load_ent=True,
        #      init_reset=False)
        # test(base=path, emb_name=name, iteration=10, worker=worker, load_weight=1.0, reset_weight=1.0, load_ent=False,
        #      load_emb=True,
        #      init_reset=False, func=fusion_func_8_2)
        test(base=path, emb_name=name, iteration=10, worker=worker, load_weight=1.0, reset_weight=1.0, load_ent=True,
             load_emb=True,
             init_reset=False, func=fusion_func_5_5)
        # test(base=path, emb_name=name, iteration=10, worker=worker, load_weight=1.0, reset_weight=1.0, load_ent=True,
        #      load_emb=True,
        #      init_reset=False, func=fusion_func_8_2)
        # test(base=path, emb_name=name, iteration=10, worker=worker, load_weight=1.0, reset_weight=1.0, load_ent=True,
        #      load_emb=True,
        #      init_reset=False, func=fusion_func_5_5)
        print("------------------------------------------------------------------------")
