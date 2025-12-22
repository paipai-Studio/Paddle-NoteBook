import os
import sys
import json
from tqdm import tqdm

try:
    work_txt = sys.argv[1]
except:
    work_txt = "x3nlp_1_test.txt"
# os.system("clear")
print(f"work_txt: {work_txt}")

# 
# 

def lookup(k, l, _k):
    for i in l:
        if i["text"] == k:
            return {
                "完全匹配": "完全匹配",
                "部分匹配": "部分匹配",
                "不匹配": "不匹配",

                "?匹配?": "不匹配",
            }.get(i["label"], "不匹配")
    print("X1")
    return "部分匹配"

w = open("test_format.txt", "w")
with open(work_txt, "r") as f:
    for j0 in tqdm(f, total=15000):
        j0 = json.loads(j0)

        try:
            j1 = os.popen(f"cat test.txt|grep {j0['text_id']}").readlines()[0]
        except:
            j1 = '{"candidate": []}'
            print("X2")
        j1 = json.loads(j1)

        # format
        j2 = {
            "text_id": j0["text_id"],
            "query": j0["query"],
            "candidate": [
                {
                    "text": i["text"],
                    "label": lookup(i["text"], j1["candidate"], j0["query"]),
                }
                for i in j0["candidate"]
            ]
        }
        w.write(f"{json.dumps(j2, ensure_ascii=False)}\n")
        # print(f"{'-'*100}\nj0: {j0}\nj1: {j1}\nj2: {j2}\n{'-'*100}")
        # break
w.close()
