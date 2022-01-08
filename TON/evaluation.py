import random
from typing import Union, List, Tuple

from treepredict import DecisionNode, buildtree, iterative_buildtree, read, classify


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

def get_accuracy(tree: DecisionNode, dataset):
    # given a decision tree and a dataset, return the number of correctly classified rows

        correct = 0
        for row in dataset:
            if classify(tree, row) == row[-1]:
                correct += 1
        return correct / len(dataset)


def cross_validation(dataset, k, agg, seed, scoref, beta, threshold):
    if seed:
        random.seed(seed)
    # If k is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(k) != int(k):
        k = int(n_rows * k)  # We need an integer number of rows
    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=k)
    # We will keep track of the scores of the different folds
    scores = []
    # For each fold
    for i in range(k):
        # We will keep track of the training and test dataset
        train = []
        test = []
        # For each row in the dataset
        for j in range(n_rows):
            # If the row is in the test dataset for this fold
            if j in test_rows:
                # Add the row to the test dataset
                test.append(dataset[j])
            else:
                # Add the row to the training dataset
                train.append(dataset[j])
        # Build the tree for the training dataset
        tree = buildtree(train, scoref, beta, threshold)
        # Compute the score for the current fold
        score = agg(get_accuracy(tree, test))
        # Add the score to the list of scores
        scores.append(score)
    # Return the mean of the scores
    return mean(scores)

# def get_accuracy(classifier, dataset):
#     raise NotImplementedError


def mean(values: List[float]):
    return sum(values) / len(values)

if __name__ == "__main__":
    #get the accuracy of the tree
    dataset = read('iris.csv')
    train, test = train_test_split(dataset, 0.5)
    tree = iterative_buildtree(train)
    print(get_accuracy(tree, test))
