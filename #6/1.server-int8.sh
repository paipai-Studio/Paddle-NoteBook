
clear
path=/home/aistudio/data/models
model=ERNIE-4.5-0.3B-Base-Paddle

python -m fastdeploy.entrypoints.openai.api_server --model $path/$model --port 8180 --metrics-port 8181 --engine-worker-queue-port 8182 --max-model-len 3072 --max-num-seqs 128 --quantization wint8
