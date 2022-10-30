from config import *
from sklearn.linear_model import LogisticRegression
from wass_loss import *
from wasserstein import *


def fit_log_reg(X, y, random_state=0):
    clf = LogisticRegression(random_state=random_state).fit(X, y)
    ic(clf.score(X, y))
    return clf


def dset_to_x_y(dset, n):
    return torch.stack([x[0] for x in dset])[0:n], torch.tensor([x[1] for x in dset])[0:n]


if __name__ == '__main__':
    ic("Logistic Regression")
