import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cost_fn(probs, damage, malus):
    bonus = 5
    return probs > malus / (bonus + malus + damage)


def bewertung(yhat, y, damage):
    res = np.zeros(yhat.shape)
    y = y[None, :]
    damage = damage[None, :]

    # Case 1: FRAUD caught
    res += ((y == 1) & (yhat == 1)) * 5
    # Case 2: False positive
    res -= ((y == 0) & (yhat == 1)) * 10
    # Case 3: FRAUD missed
    res -= ((y == 1) & (yhat == 0)) * damage
    return res.sum(axis=1)


def find_params(probs, pred_damage, y, damage):
    plothist(probs)
    n = len(y)
    mali = np.arange(1, 200)
    malus = np.tile(mali, (n, 1)).T
    yhat = cost_fn(probs, pred_damage, malus)
    res = bewertung(yhat, y, damage)
    df = pd.DataFrame({"malus": mali, "bewertung": res})
    df.to_csv("bew.csv")
    idx = np.argmax(res)
    return mali[idx]


def plothist(d):
    # use log scale
    plt.hist(d, bins=20, log=True)
    plt.savefig("hist.png")


def find_optimal_threshhold(probs, y, damage):
    threshold = np.linspace(0.1, 1.0, 100)
    threshold_grid = np.tile(threshold, (len(y), 1)).T
    preds = (probs > threshold_grid).astype(int)
    res = bewertung(preds, y, damage)
    idx = np.argmax(res)
    threshold = threshold[idx]
    return threshold
