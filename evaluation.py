import random
from typing import Union, List
from treepredict import gini_impurity
from treepredict import iterative_buildtree

from treepredict import classify, read


def train_test_split(dataset, test_size: Union[float, int], seed=None):
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(test_size) != int(test_size):
        test_size = int(n_rows * test_size)  # We need an integer number of rows

    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=test_size)

    test = [row for (i, row) in enumerate(dataset) if i in test_rows]
    train = [row for (i, row) in enumerate(dataset) if i not in test_rows]

    return train, test


def get_accuracy(classifier, dataset):
    correct = 0
    for row in dataset:
        c=classify(classifier, row[:-1])
        if c == row[-1]:
            correct += 1
    return correct / len(dataset)


def mean(values: List[float]):
    return sum(values) / len(values)


def cross_validation(dataset, k, agg, seed, scoref, beta, threshold):
    folds = train_test_split(dataset, test_size=1 / k, seed=seed)
    scores = []
    for fold in folds:
        train = [row for row in folds if row != fold]
        classifier = iterative_buildtree(train)
        score = scoref(classifier, fold)
        scores.append(score)
    return scores

h,dataset=read('iris.csv')
# train,test=train_test_split(dataset, 0.5)
# print(get_accuracy(iterative_buildtree(train), test))
print(cross_validation(dataset, 5, gini_impurity, None, gini_impurity, 0.5, 0.5))