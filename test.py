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


path_r_1 = "dataset/industry/rel_triples_1"
path_a_1 = "dataset/industry/attr_triples_1"

path_r_2 = "dataset/industry/rel_triples_2"
path_a_2 = "dataset/industry/attr_triples_2"

kg1 = construct_kg(path_r_1, path_a_1, "KG1")
kg2 = construct_kg(path_r_2, path_a_2, "KG2")

kgs = KGs(kg1=kg1, kg2=kg2, iteration=3)
kgs.run()
kgs.output_alignment_result()
kgs.store_params()


# config = PARISConfig()
# print(config.get_similarity("Saturday", "Sundays"))
