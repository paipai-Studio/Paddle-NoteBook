import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

os.system("clear")


# 1.Data Preparation
# CUT TRAIN && EVAL DATA TO JSONL
data = pd.read_csv(
    "x3nlp_1_train.txt", sep="\t", header=None, names=["text"]
)
d1, d2 = train_test_split(data, test_size=0.2)
print(d1.shape, d2.shape)

for _name, _data in [
    ["SFT-train.jsonl", d1],
    ["SFT-eval.jsonl", d2],
]:
    with open(_name, "w") as f:
        for i in _data["text"]:
            j = json.loads(i)
            j1 = {
                "query": j["query"], 
                "candidate": [{"text": ij["text"]} for ij in j["candidate"]]
            }
            j2 = {
                "query": j["query"], 
                "candidate": j["candidate"]
            }
            j3 = {
                "src": [json.dumps(j1, ensure_ascii=False)],
                "tgt": [json.dumps(j2, ensure_ascii=False)],
            }
            # print(j1, "\n\n", j2, "\n\n", j3)
            f.write(json.dumps(j3, ensure_ascii=False)+"\n")
            # break


p1 = "data/models/ERNIE-4.5-21B-A3B-Base-LORA"

# 2.Supervised Fine-tuning
os.system(f"rm -rf {p1}")
os.system("erniekit train SFT-run_sft_lora_8k.yaml")


# 3.Weight Merging
os.system(f"rm -rf {p1}/export")
os.system("erniekit export SFT-run_export.yaml lora=True")


# 4.Load Model To Server
os.system(f"python -m fastdeploy.entrypoints.openai.api_server --model {p1}/export --port 8180 --metrics-port 8181 --engine-worker-queue-port 8182 --max-model-len 1024 --max-num-seqs 128")

