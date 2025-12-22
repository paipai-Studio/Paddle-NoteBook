import os
import sys
import json
import openai
import pandas as pd

# Import
from pandarallel import pandarallel
# Initialization
pandarallel.initialize(nb_workers=64, progress_bar=True)

try:
    work_txt = sys.argv[1]
except:
    work_txt = "x3nlp_1_test.txt"
# os.system("clear")
print(f"work_txt: {work_txt}")

host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

_system = {
    "role": "system", 
    "content": """
你是一个地址匹配专家，你需要判断输入的json中，query和candidate下的每一个text都是否为同一个地址信息，如果匹配正确则label为"完全匹配"，如果匹配不正确则label为"不匹配"，如果只有部分匹配则label为"部分匹配"，返回结果严格按照json格式。

输入：
{"query": "江苏省南京市清水亭东路9号金域蓝湾15幢", "candidate": [{"text": "江宁区万科金域蓝湾15栋"}, {"text": "江苏省南京市清水亭东路9号"}, {"text": "新水泥路666号重工数控工业园"}]}

输出：
{"query": "江苏省南京市清水亭东路9号金域蓝湾15幢", "candidate": [{"text": "江宁区万科金域蓝湾15栋", "label": "完全匹配"}, {"text": "江苏省南京市清水亭东路9号", "label": "部分匹配"}, {"text": "新水泥路666号重工数控工业园", "label": "不匹配"}]}
"""
}


def get(_c, _i=""):
    _jc = json.loads(_c)
    _jc_id = _jc["text_id"]
    _c = json.dumps({
        "query": _jc["query"],
        "candidate": [{"text": i["text"]} for i in _jc["candidate"]],
    }, ensure_ascii=False)
    # print(_c)

    response = client.chat.completions.create(
        model="null",
        messages=[_system, {"role": "user", "content": _c}],
        stream=False,

        extra_body={
            "penalty_score": 0.0,   # 关闭惩罚，避免干扰匹配
        },
        max_completion_tokens=512,  # 地址通常较短，
        temperature=0.01,           # 完全确定性输出，避免随机性
        top_p=1.0,                  # 关闭核采样（与temperature=0冲突，可保留）
        frequency_penalty=0.0,      # 不惩罚重复（地址可能有重复元素）
        presence_penalty=0.0,       # 不惩罚存在性（关键信息需保留）

        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "candidate": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string", "enum": ["完全匹配", "部分匹配", "不匹配"]}
                            },
                            "required": ["text", "label"]
                        }
                    }
                },
                "required": ["query", "candidate"]
            }
        }
    )

    try:
        _j = json.loads(
            response.choices[0].message.content.replace("`","").replace("json","")
        )
    except Exception as e:
        # print(f"\n \033[1;36m ERROR {_i}:\033[0m \n{response.choices[0].message.content[:100]}\n{e}", )
        _j = json.loads(_c)
    finally:
        _j["text_id"] = _jc_id

        for _ji in _j["candidate"]:
            if "label" not in _ji:
                _ji.update({"label": "?匹配?"})
        return json.dumps(_j, ensure_ascii=False)


t = """
{"text_id": "2b51366fdd6c620a3f54b520a8ebc5e5", "query": "兴东街道铁成佳园农贸大市场2厅", "candidate": [{"text": "沙河镇街9-1号铁成佳园农贸大市场", "label": "完全匹配"}, {"text": "沙河镇街11号铁成佳园农贸大市场(东南1门)", "label": "不匹配"}, {"text": "宏安路与沙河镇街交叉口北50米铁成佳园农贸大市场(东南2门)", "label": "部分匹配"}, {"text": "37县道佳园农场", "label": "部分匹配"}, {"text": "杭州大道农贸大市场", "label": "不匹配"}]}
"""
print(f"Test:\n{t}\n{get(t)}\nStart:\n")
# raise "Test"

data = pd.read_csv(
    work_txt, sep="\t", header=None, names=["text"]
)
# parallel_apply
data["result"] = data["text"].parallel_apply(get)

with open("test.txt", "w") as f:
    for i in data["result"]:
        f.write(f"{i}\n")

