import numpy as np


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
    n = len(y)
    mali = np.arange(1, 200)
    malus = np.tile(mali, (n, 1)).T
    yhat = cost_fn(probs, pred_damage, malus)
    res = bewertung(yhat, y, damage)
    idx = np.argmax(res)
    return mali[idx]
