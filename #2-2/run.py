
import os

os.system("""
cd data/data298599
ls -l && rm -rf images && unzip -q images.zip && clear
ls -l images/*.png|wc -l
""")

os.system("""
sh build.sh
""")

import paddle
import paddlex
print(paddle.__version__, paddlex.__version__)




import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from cmaes import CMA
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="object_detection")


# 参数定义
# 图像大小
N = 5
# 图像尺寸
S = N * N * 3
# 迭代次数和种群大小
R, P = 100, 10


Ipath = "data/data298599/images"
for Iname in tqdm(os.listdir(Ipath), total=1000):
    if Iname.endswith(".png"):
        base = cv.imread(f"{Ipath}/{Iname}")
        print(Iname, base.shape)

        def floss(x, save=False):
            _x = np.resize(x.reshape((N, N, 3)), (500, 500, 3))
            # cv.imwrite(f"result/output/X{Iname}", _x)
            cv.imwrite(f"result/images/{Iname}", base + _x)
            
            n, score = 0, 0
            output = pipeline.predict(f"result/images/{Iname}")
            for res in output:
                if save:
                    res.save_to_img("result/output/")
                for box_res in res["boxes"]:
                    n += 1
                    score += box_res["score"]
            return np.abs(_x).mean(), score/n if n > 0 else 0

        # 定义CMA优化器
        OPT = CMA(
            mean=np.zeros(S),
            sigma=10, 
            bounds=np.array([[-255, 255] for _ in range(S)]), 
            population_size=P
        )

        x_min, loss_min = None, np.inf
        for rI in range(R):
            solutions = []
            for rJ in range(P):
                x = OPT.ask()
                l1, l2 = floss(x)
                # 定义loss函数
                loss = l1 * 0.1 + l2 * 0.9
                solutions.append((x, loss))

                _r = "\r"
                if loss_min > loss:
                    loss_min = loss
                    x_min = x

                    # floss(x, save=True)
                    _r = "\tMIN\n"

                print(f"\t#{rI:5d}/{rJ:5d}\tloss:{loss:.6f}\tl1:{l1:.6f}\tl2:{l2:.6f}{_r}", end="")
            OPT.tell(solutions)
        floss(x_min, save=True)
        # break

os.system("""
cd result
rm -rf images/.ipynb_checkpoints
zip -r images.zip images/

ls -l -h images.zip
md5sum images.zip
""")

