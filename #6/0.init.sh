
pip install pandarallel

clear
path=/home/aistudio/data/models
model=ERNIE-4.5-0.3B-Base-Paddle

echo $model
rm -rf $path/$model
aistudio download --model PaddlePaddle/$model --local_dir $path/$model
ls -l $path/$model
