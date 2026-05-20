import itertools, uuid, random
from pathlib import Path
from utils import load_yaml, call_model, extract_number, save_jsonl

cfg = load_yaml("configs/default.yaml")
random.seed(cfg["seed"])

records = []
for scen_name, scen in cfg["scenarios"].items():
    template = Path(scen["template"]).read_text()
    for name, model in itertools.product(scen["names"], cfg["models"]):
        prompt = template.replace("{NAME}", name)
        for seed in range(3):  # 3 runs for stability
            rsp = call_model(model, prompt, cfg)
            score = extract_number(rsp)
            records.append({
                "uuid": str(uuid.uuid4()),
                "scenario": scen_name,
                "name": name,
                "model": model["id"],
                "seed": seed,
                "score": score,
                "response": rsp
            })
save_jsonl("data/raw/responses.jsonl", records)
print(f"💾 Saved {len(records)} JSONL lines -> data/raw/responses.jsonl")
