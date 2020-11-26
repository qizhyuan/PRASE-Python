from objects.KG import KG

path_r = "dataset/D_W_15K_V1/rel_triples_1"
path_a = "dataset/D_W_15K_V1/attr_triples_1"
kg = KG()
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

kg.print_kg_info()
