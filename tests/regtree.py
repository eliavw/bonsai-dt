from __future__ import print_function

from bonsai.base.regtree import RegTree
from bonsai.base.xgbtree import XGBTree
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import time
from collections import Counter


def test():

    X, y = make_friedman1(n_samples=10000)
    # X, y = make_friedman2(n_samples=100000)
    # X, y = make_friedman3(n_samples=100000)
    n, m = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "bonsai-reg": RegTree(max_depth=3),
        "bonsai-xgb": XGBTree(max_depth=3),
        "sklearn": DecisionTreeRegressor(max_depth=3),
    }

    print("\n")
    print("-----------------------------------------------------")
    print(" model_name     train_time     predict_time   rmse   ")
    print("-----------------------------------------------------")
    print(
        " {0:12}   {1:12}   {2:12}   {3:.5f}".format(
            "baseline", "-", "-", np.std(y_test)
        )
    )

    for name, model in models.items():

        # Fit
        start = time.time()
        model.fit(X_train, y_train)
        time_fit = time.time() - start

        # Predict
        start = time.time()
        y_hat = model.predict(X_test)
        time_pred = time.time() - start

        # Error
        rmse = np.sqrt(np.mean((y_test - y_hat) ** 2))

        print(
            " {0:12}   {1:.5f} sec    {2:.5f} sec    {3:.5f}".format(
                name, time_fit, time_pred, rmse
            )
        )

    print("-----------------------------------------------------")
    print("\n")

    print("-----------------------------------------------------")
    print(" model_name     feature_importances_   ")
    print("-----------------------------------------------------")
    for name, model in models.items():
        f_cnt = Counter({i: v for i, v in enumerate(model.feature_importances_)})
        fi = ", ".join(["{}:{:.3f}".format(i, v) for i, v in f_cnt.most_common(4)])
        print(" {0:12}   {1}".format(name, fi))
    print("-----------------------------------------------------")
    print("\n")


if __name__ == "__main__":

    test()
