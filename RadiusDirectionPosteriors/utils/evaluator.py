import torch
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import os

# sns.set_theme()
 
class Evaluator():
    def __init__(self):
        self.dict = {"train_acc":[], "test_acc": [],
                    "train_nll": [], "test_nll":[],
                    "train_kld": [], "test_kld": []}

    def eval(self, **dic):
        for key in dic.keys():
            self.dict[key].append(dic[key])

        # print(self.dict)
    
    def plot(self, dir):
        sns.lineplot(data=self.dict["train_acc"], label="train")
        sns.lineplot(data=self.dict["test_acc"], label="test")
        plt.set_ylabel("Accuracy")
        plt.set_xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(dir, "acc.png"))
        plt.clf()

        sns.lineplot(data=self.dict["train_nll"], label="train")
        sns.lineplot(data=self.dict["test_nll"], label="test")
        plt.set_ylabel("NLL")
        plt.set_xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(dir, "nll.png"))
        plt.clf()

        sns.lineplot(data=self.dict["train_kld"], label="train")
        sns.lineplot(data=self.dict["test_kld"], label="test")
        plt.set_ylabel("KLD")
        plt.set_xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(dir, "kld.png"))
        plt.clf()



