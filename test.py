from objects.KG import KG
from objects.KGs import KGs


def construct_kg(path_r, path_a, name=None):
    kg = KG(name=name)
    with open(path_r, "r", encoding="utf-8") as f:
        for line in f.readlines():
            params = str.strip(line).split(sep='\t')
            assert len(params) == 3
            h, r, t = params[0].strip(), params[1].strip(), params[2].strip()
            kg.insert_relation_tuple(h, r, t)

    with open(path_a, "r", encoding="utf-8") as f:
        for line in f.readlines():
            params = str.strip(line).split(sep='\t')
            assert len(params) == 3
            e, a, v = params[0].strip(), params[1].strip(), params[2].strip()
            kg.insert_attribute_tuple(e, a, v)
    kg.calculate_functionality()
    kg.print_kg_info()
    return kg


path_r_1 = "dataset/EN_DE_15K_V1/rel_triples_1"
path_a_1 = "dataset/EN_DE_15K_V1/attr_triples_1"

path_r_2 = "dataset/EN_DE_15K_V1/rel_triples_2"
path_a_2 = "dataset/EN_DE_15K_V1/attr_triples_2"

kg1 = construct_kg(path_r_1, path_a_1, "KG1")
kg2 = construct_kg(path_r_2, path_a_2, "KG2")

kgs = KGs(kg1=kg1, kg2=kg2, iteration=1, ent_lite_candidate_num=2, rel_attr_candidate_num=3, output_threshold=0, refine_threshold=0.1)

# kgs.run()
# kgs.load_params()

path_test = "dataset/industry/ent_links"
with open(path_test, "r", encoding="utf-8") as f:
    num = 2000
    for line in f.readlines():
        params = str.strip(line).split(sep='\t')
        assert len(params) == 2
        e, a = params[0].strip(), params[1].strip()
        kgs.insert_ent_tuple(e, a, 1.0)
        num -= 1
        if num <= 0:
            break

kgs.run()
kgs.output_alignment_result()
# kgs.store_params()
