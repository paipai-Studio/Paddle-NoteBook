
pip install pandarallel

clear
path=/home/aistudio/data/models
model=ERNIE-4.5-21B-A3B-Base-Paddle

echo $model
rm -rf $path/$model
aistudio download --model PaddlePaddle/$model --local_dir $path/$model
ls -l $path/$model

python -m fastdeploy.entrypoints.openai.api_server --model $path/$model --port 8180 --metrics-port 8181 --engine-worker-queue-port 8182 --max-model-len 1024 --max-num-seqs 128
