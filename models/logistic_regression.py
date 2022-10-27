from config import *
from sklearn.linear_model import LogisticRegression


def fit_log_reg(X, y, random_state=0):
    clf = LogisticRegression(random_state=random_state).fit(X, y)
    ic(clf.score(X, y))
    return clf


if __name__ == '__main__':
    ic("Logistic Regression")
