import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

 
class Evaluator():
    def __init__(self):
        self.dict = {"train_acc":[], "test_acc": [],
                    "train_nll": [], "test_nll":[],
                    "train_kld": [], "test_kld": []}

    def eval(self, **dic):
        for key in dic.keys():
            self.dict[key].append(dic[key])

    def save(self, dir):
        for key in self.dict.keys():
            self.dict[key] = np.array(self.dict[key])
        arr = np.array([self.dict[key] for key in ["train_acc", "test_acc",
                    "train_nll", "test_nll",
                    "train_kld", "test_kld"]])
        np.save(os.path.join(dir, "logs.npy"), arr)
    
    def plot(self, dir):
        sns.lineplot(data=self.dict["train_acc"], label="train")
        sns.lineplot(data=self.dict["test_acc"], label="test")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(dir, "acc.png"))
        plt.clf()

        sns.lineplot(data=self.dict["train_nll"], label="train")
        sns.lineplot(data=self.dict["test_nll"], label="test")
        plt.ylabel("NLL")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(dir, "nll.png"))
        plt.clf()

        sns.lineplot(data=self.dict["train_kld"], label="train")
        sns.lineplot(data=self.dict["test_kld"], label="test")
        plt.ylabel("KLD")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(dir, "kld.png"))
        plt.clf()



