import os
import sys
import json
import openai
import pandas as pd
from tqdm import tqdm

os.system("clear")

# Import
from pandarallel import pandarallel
# Initialization
pandarallel.initialize(nb_workers=64, progress_bar=True)

host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

_system = {
    "role": "system", 
    "content": """
你是一个自然语言处理专家，现在需要进行问题匹配，不匹配返回0，匹配返回1，

输入：
{"A": "婴儿吃什么蔬菜好", "B": "婴儿吃什么绿色蔬菜好"}

输出：
{"result": 0}

严格按照输出的json格式。
"""
}
_E1, _E2 = 0, 0


def get(_c):
    response = client.chat.completions.create(
        model="null",
        messages=[
            _system, 
            {
                "role": "user", 
                "content": json.dumps(_c, ensure_ascii=False)
            }
        ],
        stream=False,
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "int"}
                },
                "required": [ 
                    "result"
                ]
            }
        }
    )

    try:
        _j = json.loads(
            response.choices[0].message.content.replace("`","").replace("json","")
        )
        _r = int(_j.get("result", 0))
    except Exception as e:
        print(f"\n \033[1;36m ERROR:\033[0m \n{response.choices[0].message.content[:100]}\n{e}", )
        _r = 0
    return _r if _r == 0 else 1


t = """
{"A": "婴儿吃什么蔬菜好", "B": "婴儿吃什么绿色蔬菜好"}
"""
print(f"""Test "{get(t)}".\n""")
# raise "Test"


data = pd.read_csv(
    "test/test.tsv", 
    sep="\t", header=None,
    names=["A", "B"],
)
print(data.shape)

data["text"] = [
    {"A": f"{_1}", "B": f"{_2}"}
    for _1, _2 in zip(data["A"], data["B"])
]
data["result"] = data["text"].parallel_apply(get)
print(data)
print(data["result"].value_counts())

with open("test/predict.csv", "w") as f:
    for i in data["result"]:
        f.write(f"{i}\n")

