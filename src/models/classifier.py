"""Downstream classification heads, trained on top of the frozen
SupCon encoder features.

Three options, matching what the original notebooks explored:
  - LinearClassifier:      a single nn.Linear layer, trained with SGD/Adam.
  - build_logreg_grid_search(): sklearn LogisticRegression + GridSearchCV.
  - build_svc_grid_search():    sklearn SVC + GridSearchCV.
"""
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def build_logreg_grid_search(cv_splits: int = 5, random_state: int = 42) -> GridSearchCV:
    """LogisticRegression + GridSearchCV (malware-analyze notebook variant)."""
    classifier = LogisticRegression(penalty="l2", solver="saga")
    param_grid = {
        "solver": ["saga", "lbfgs"],
        "penalty": ["l1", "l2"],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    return GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "auc_ovr": "roc_auc_ovr",
        },
        refit="f1_macro",
    )


def build_svc_grid_search(cv_splits: int = 5, random_state: int = 42) -> GridSearchCV:
    """SVC + GridSearchCV (mal-api-2019 notebook variant).

    Uses balanced class weights and OvR decision function to handle
    multi-class malware-family classification.
    """
    classifier = SVC(decision_function_shape="ovr", class_weight="balanced", probability=True)
    param_grid = {
        "C": [0.01, 0.8, 1, 1.2, 1.5],
        "kernel": ["linear", "rbf"],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    return GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "auc_ovr": "roc_auc_ovr",
        },
        refit="f1_macro",
    )
