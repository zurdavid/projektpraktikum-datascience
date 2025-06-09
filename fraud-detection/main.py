from pathlib import Path

import fraud_detection as fd

path = Path("./data/transformed_label_and_damage.parquet")


def compare():
    X, _ = fd.data_loader.load_data_np(path)
    clf = fd.neuralnets.train_nn.getNN(X.shape[1])

    models = [
        # ("NeuralNet", clf),
        ("LightGBM", fd.models.get_lgmb()),
        ("XGBoost", fd.models.get_xgb(10)),
        ("XGBoost Plus", fd.models.models.get_xgb_clf_with_reg(10)),
    ]

    model_metrics = fd.model_comparison.compare_models(models, path, n_splits=5)
    df = pd.DataFrame(model_metrics)


def train_single():
    X, y = fd.data_loader.load_data_np(path)
    clf = fd.neuralnets.train_nn.getNN(X.shape[1])
    # clf = fd.models.get_lgmb()

    # clf = fd.models.models.get_xgb_plus(10)
    fd.model_comparison.train_model(clf, X, y)


if __name__ == "__main__":
    train_single()
    # compare()
