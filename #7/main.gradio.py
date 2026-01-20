
import os
os.system("pip install erniebot==0.5.0")
os.system("pip install gradio==4.26.0 --user")
os.system("pip install dotenv==0.9.9")
os.system("pip install openai==2.14.0")

import base64
import uuid
import time
import pickle
import random
import warnings
import datetime
import cv2 as cv
import numpy as np
import pandas as pd
import paddle as pdl
import gradio as gr
from openai import OpenAI

# 通过.env配置
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

from paras import PaiPai, iDataset, jDataset, COLS, _v2k, yN, _k2v_num, Mcols, McolsN, iMessages

# python3 -c "import uuid;print(str(uuid.uuid1()))"
UUID = str(uuid.uuid1())

SEED = 1024
pdl.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

encoder = PaiPai()
encoder.set_state_dict(pdl.load("./modl/_s5.5066_t5.4835_v5.5066.mdl"))
encoder.eval()

client = OpenAI(
    api_key=API_KEY, 
    base_url="https://aistudio.baidu.com/llm/lmapi/v3",
)

wrkp = "./work"

_HEAD = """<p align="center"><b>EDA, 眼部疾病诊断AI助手</b></p>"""
_ING1 = """<p align="center" style="color: red;font-weight: bold;">EDA诊断中, 请稍等...</p>"""
_ING2 = """<p><i>眼底萤光血管造影 (FFA) 是由手臂静脉注入萤光剂, 通过眼底相机照出黄斑部及视网膜血管的变异情形, 可作为医师诊断及后续治疗的参考辅助, 也能让文心化身AI眼科医生。</i></p>"""
_ENDS = """<p align="center" style="color: green;">EDA诊断结束</p>"""


_NC, _P, _M, _E = {"C": 0}, {}, {}, {}


getDT = lambda: datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')


def threshold(img):
    try:
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
        ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)  
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return cv.drawContours(img, contours, -1, (0, 255, 0), 1)
    except:
        return img


def fmodel(x, progress=gr.Progress()):
    for _task in progress.tqdm(range(4)):
        if _task == 0:
            _k = np.sum(x)

        if _task == 1:
            if _k not in _P:
                img = x.copy()
                if img.shape == (434, 768, 3):
                    img = img[:384,:,:,].copy()
                elif img.shape == (409, 384, 3):
                    img.resize((434, 384, 3))
                    img = cv.hconcat([img, img])
                    img = img[:384,:,:,].copy()
                elif img.shape == (434, 384, 3):
                    img.resize((434, 384, 3))
                    img = cv.hconcat([img, img])
                    img = img[:384,:,:,].copy()
                elif img.shape == (306, 512, 3):
                    _img1 = img[:,:256,:].copy()
                    _img2 = img[:,256:,:].copy()
                    _img1 = cv.resize(_img1[:256,:,:,].copy(), (384, 384))
                    _img2 = cv.resize(_img2[:256,:,:,].copy(), (384, 384))
                    img = cv.hconcat([_img1, _img2])
                elif img.shape == (281, 256, 3):
                    img.resize((434, 384, 3))
                    img = cv.hconcat([img, img])
                    img = img[:384,:,:,].copy()
                else:
                    # img.resize((384, 768, 3))
                    _img1 = np.random.randint(0, 1, (384, 768, 3))
                    _l1, _l2 = min([384, img.shape[0]]), min([768, img.shape[1]])
                    _img1[:_l1,:_l2,:,] = img[:_l1,:_l2,:,]
                    img = _img1.copy()
                
                _P[_k] = img
                warnings.warn(f"{getDT()} _P.SET {_k}")
            else:
                time.sleep(0.618)
                img = _P[_k]
                warnings.warn(f"{getDT()} _P.GET {_k}")

        if _task == 2:
            if _k not in _M:
                _x1, _x2 = jDataset(img)
                _y = encoder(_x1, _x2).cpu().numpy()[0]

                _result = "我的眼睛 "
                for _n in range(yN):
                    _m = COLS[_n]
                    _max = _k2v_num[_m] - 1
                    _st, _ed = 32*_n, 32*(_n+1)

                    _yn = np.argmax(_y[_st:_ed], axis=0)
                    _yn = np.where(_yn > _max, _max, _yn).item()
                    _result += f"{_m} {min(_y[_st:_ed][_yn], 0.9999):.2%}可能是 {_v2k[_m][_yn]}, "
                
                _M[_k] = _result
                warnings.warn(f"{getDT()} _M.SET {_k}")
            else:
                time.sleep(0.618)
                _result = _M[_k]
                warnings.warn(f"{getDT()} _M.GET {_k}")

        if _task == 3:
            _NC["C"] += 1
            warnings.warn(f"{getDT()} JOB.COUNT {_NC['C']:08d}")
    
    return threshold(img), f"{time.time()}|{_result}", f"{_ING1}\n{_ING2}"



# 读取图片文件并转换为 Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ferniebot(_i, _k):
    # 临时保存 并进行压缩处理
    cv.imwrite(f"{wrkp}/working.jpg", cv.resize(_i, (24, 24)))
    _i = encode_image(f"{wrkp}/working.jpg")

    _k = _k.split("|")[1]

    if _k not in _E:
        try:
            response = client.chat.completions.create(
                model="ernie-5.0-thinking-preview",
                messages=iMessages(_i, _k),

                stream=False,
                extra_body={
                    "web_search": {
                        "enable": True
                    }
                },
                max_completion_tokens=65536,
            )
            _result = response.choices[0].message.content
        except:
            _result = "Sorry."
        
        _E[_k] = _result
        warnings.warn(f"{getDT()} _E.SET {_k[:50]}")
    else:
        time.sleep(3.618 + random.random())
        _result = _E[_k]
        warnings.warn(f"{getDT()} _E.GET {_k[:50]}")
    
    return _ENDS, _result


with gr.Blocks(
        title="EDA", 
        theme=gr.themes.Glass(), # Default/Soft/Monochrome/Glass
    ) as APP:

    with gr.Tab("EDA"):
        with gr.Column():
            m1 = gr.Markdown(_HEAD)
            i1 = gr.Image(
                show_label=False, 
                sources=["upload"],
            )
            eg = gr.Examples(
                examples=[f"{wrkp}/{i}.jpg" for i in range(4)], # 16
                label="请上传或选择一张FFA/眼底萤光血管造影:",
                inputs=i1, 
                outputs=i1,
                fn=lambda x: x,
                run_on_click=True,
                examples_per_page=4,
            )
        t1 = gr.Textbox(visible=False)
        b1 = gr.Button(value="开始诊断", size="sm")
        with gr.Row(variant="panel"):
            m4 = gr.Markdown()

        b1.click(inputs=i1, outputs=[i1, t1, m1], fn=fmodel)
        t1.change(inputs=[i1, t1], outputs=[m1, m4], fn=ferniebot)


    with gr.Tab("About"):
        gr.Markdown(
        f"""
        ### EDA(Eye Disease Diagnostic AI Assistant)
        """)
        with gr.Row(variant="panel"):
            gr.Markdown(
            f"""
            * 您只需要上传一张FFA/眼底萤光血管造影，便可以获得清晰的诊断分析，让EDA眼部疾病诊断AI助手成为你的线上诊断专家。只需一键，就能得到专业的眼部疾病诊断，就像按下快门捕捉瞬间一样简单，让您的眼部健康问题得到迅速解答。这款应用不仅能够利用最新的医疗图像识别技术，为你提供快速、准确的眼部疾病诊断，还能够为你提供深入的疾病分析和个性化的治疗建议。
            """)
        url1 = '<a href="https://aistudio.baidu.com/personalcenter/thirdview/979775">Baidu</a>'
        url2 = '<a href="https://github.com/IvanaXu">GitHub</a>'
        url3 = '<a href="mailto:eda@tangjt.cn">eda@tangjt.cn</a>'
        gr.Markdown(
        f"""        
        THANKS ALL !
        <p align="right"><small>{url1} / {url2} , {url3}</small></p>
        <p align="right"><small>{UUID}-030</small></p>
        """)


APP.queue()
APP.launch(
    # server_name="0.0.0.0", 
    # server_port=7860,
    # ssl_keyfile="./data/ssl/key.pem",
    # ssl_certfile="./data/ssl/cert.pem",
)

