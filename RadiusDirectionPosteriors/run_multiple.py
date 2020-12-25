import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="MNIST")
parser.add_argument("--model", type=str, default="LeNet5")
args = parser.parse_args()

for lr in [0.01, 0.001, 0.0001]:
    for optim in ["SGD", "Adam", "AMSGrad"]:
        for bs in [16, 32, 64, 128]:
            os.system(f"sbatch --gres=gpu:1 --mem=6G --time=3:0:0 run.sh {lr} {bs} {optim} {args.data} {args.model}")
