# Modelle

## Klassifikation

### XGBoost

```python
- package: `xgboost-cpu`

XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective="binary:logistic",
)
```

### LightGBM
- package: `lightgbm`

```python
LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=45,
)
```

### CatBoost
- package: `catboost`

```python
CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    verbose=0,
    random_seed=42,
)
```

### Logicstic Regression
- package: `sklearn`
- verwendet mit `RobustScaler`

```python
LogisticRegression(
    max_iter=5000, 
    class_weight="balanced", 
    solver="saga"
)
```


### Decision Tree
- package: `sklearn`

```python
DecisionTreeClassifier(
    max_depth=8,
    class_weight="balanced",
)
```


### Decision Tree
- package: `sklearn`

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight="balanced",
)
```


### Neuronales Netzwerk
- Feedforward Neural Network
- package `pytorch`

```python
inner_layers = [64, 32]
dropout = 0.4
loss_fn = FocalLoss(alpha=0.2, gamma=1.5, pos_weight=pos_weight)

optimizer = Adam
scheduler = OneCycleLR(max_lr=1e-2)
batch-size = 256
trainin-epochs: 15
```
