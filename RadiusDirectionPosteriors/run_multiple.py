import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="MNIST")
parser.add_argument("--model", type=str, default="LeNet5")
args = parser.parse_args()

for lr in [0.01, 0.001, 0.0001]:
    for optim in ["SGD", "Adam", "AMSGrad"]:
        for bs in [16, 32, 64, 128]:
            os.system(f"sbatch --gres=cpu:1 --mem=8G --time=3:0:0 run.sh {lr} {bs} {optim} {data} {model}")
for model in models:
    if model != args.algo:
        continue
    for variant in extras[model]:
        if variant is None:
            ex = ''
        else:
            ex = ' --'.join(variant)
            ex = '--'+ex
        for seed in range(1, seeds+1):
            os.system(f"sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=168:0:0 run.sh {model} {args.data} {seed} \'{ex}\'")
