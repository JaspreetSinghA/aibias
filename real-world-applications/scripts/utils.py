import os, json, yaml, re
from dotenv import load_dotenv
load_dotenv()

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def extract_number(text):
    import re
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group()) if m else None

def call_model(model_cfg, prompt, cfg):
    prov = model_cfg["provider"]
    if prov == "openai":
        import openai
        client = openai.OpenAI()
        rsp = client.chat.completions.create(
            model=model_cfg["id"],
            messages=[{"role":"user","content":prompt}],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"]
        )
        return rsp.choices[0].message.content.strip()
    elif prov == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        rsp = client.messages.create(
            model=model_cfg["id"],
            messages=[{"role":"user","content":prompt}],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"]
        )
        return rsp.content[0].text.strip()
    elif prov == "groq":
        import groq
        client = groq.Groq()
        rsp = client.chat.completions.create(
            model=model_cfg["id"],
            messages=[{"role":"user","content":prompt}],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"]
        )
        return rsp.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown provider: {prov}")
