import json
import _sep1

with open("gcg-paper/external-benchmarks/jsonschemabench/jsonschemabench/maskbench/data/Glaiveai2K---search_jobs_ece2422b.json") as f:
    data = json.load(f)

schema = data["schema"]
schema_str = json.dumps(schema)
ebnf = _sep1.json_schema_to_ebnf_py(schema_str)
print(ebnf)
