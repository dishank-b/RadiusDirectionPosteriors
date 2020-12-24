import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calib(pred, target, prob):
    eps = 0.05
    prev_p=0.0
    acc = []
    conf = []
    for p in np.arange(eps,1.0+eps, eps):
        indx = np.logical_and(prob>prev_p, prob<=p)
        if indx.sum()==0:
            continue
        acc_p = float((pred[indx]==target[indx]).sum())/indx.sum()
        conf_p = prob[indx].sum()/indx.sum()
        prev_p = p 

        acc.append(acc_p)
        conf.append(conf_p)
    
    return np.array(acc), np.array(conf)

 
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

    def plot_calib(self, dir, acc, conf):
        np.save(os.path.join(dir, "logs_calib.npy"), np.array([acc, conf]))
        plt.plot(conf, acc)
        plt.ylabel("Actual Frequency")
        plt.xlabel("Predicted Confidence")
        plt.savefig(os.path.join(dir, "calib.png"))
        plt.clf()






