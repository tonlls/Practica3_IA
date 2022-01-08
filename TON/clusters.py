from typing import Tuple, List
from math import sqrt


def readfile(filename: str) -> Tuple[List, List, List]:
    headers = None
    row_names = list()
    data = list()

    with open(filename) as file_:
        for line in file_:
            values = line.strip().split("\t")
            if headers is None:
                headers = values[1:]
            else:
                row_names.append(values[0])
                data.append([float(x) for x in values[1:]])
    return row_names, headers, data


# .........DISTANCES........
# They are normalized between 0 and 1, where 1 means two vectors are identical
def euclidean(v1, v2):
    return sqrt(sum([(v1[i] - v2[i])**2 for i in range(len(v1))]))

def euclidean_squared(v1, v2):
    return euclidean(v1, v2)**2

def pearson(v1, v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)
    # Sums of squares
    sum1sq = sum([v**2 for v in v1])
    sum2sq = sum([v**2 for v in v2])
    # Sum of the products
    products = sum([a * b for (a, b) in zip(v1, v2)])
    # Calculate r (Pearson score)
    num = products - (sum1 * sum2 / len(v1))
    den = sqrt((sum1sq - sum1**2 / len(v1)) * (sum2sq - sum2**2 / len(v1)))
    if den == 0:
        return 0
    return 1 - num / den


# ........HIERARCHICAL........
class BiCluster:
    def __init__(self, vec, left=None, right=None, dist=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = dist

def hcluster(rows, distance=pearson):
    distances = {}  # Cache of distance calculations
    currentclustid = -1  # Non original clusters have negative id

    # Clusters are initially just the rows
    clust = [BiCluster(row, id=i) for (i, row) in enumerate(rows)]

    """
    while ...:  # Termination criterion
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                distances[(clust[i].id, clust[j].id)] = ...

            # update closest and lowestpair if needed
            ...
        # Calculate the average vector of the two clusters
        mergevec = ...

        # Create the new cluster
        new_cluster = BiCluster(...)

        # Update the clusters
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(new_cluster)
    """

    return clust[0]

def printclust(clust: BiCluster, labels=None, n=0):
    # indent to make a hierarchy layout
    indent = " " * n
    if clust.id < 0:
        # Negative means it is a branch
        print(f"{indent}-")
    else:
        # Positive id means that it is a point in the dataset
        if labels == None:
            print(f"{indent}{clust.id}")
        else:
            print(f"{indent}{labels[clust.id]}")
    # Print the right and left branches
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n+1)


# ......... K-MEANS ..........
def kcluster(rows, distance, k=4, num_iterations=20):
    #return a tuple containing (the centroids found, the sum of the distances of each point to its centroid)
    # Initialize k randomly placed centroids
    centroids = [r[:] for r in rows[:k]]
    lastmatches = None
    for t in range(num_iterations):
        print(f"Iteration {t}")
        # Create a list of clusters
        clusters = [list() for _ in range(k)]
        # Each point is added to the closest cluster
        for row in rows:
            # Find which cluster the point belongs to
            minidx = min([(i, distance(row, c)) for i, c in enumerate(centroids)], key=lambda t: t[1])[0]
            # Add the point to that cluster
            clusters[minidx].append(row)
        # Calculate the new centroids
        for i in range(k):
            centroids[i] = [mean(x) for x in zip(*clusters[i])]
        # If the results are the same as last time, this is complete
        if lastmatches == clusters:
            break
        lastmatches = clusters
    return centroids, clusters

def mean(values: List[float]):
    return sum(values) / len(values)

if __name__ == "__main__":
    # Read the data
    row_names, headers, data = readfile("iris.csv")
    # Cluster the data
    clust = hcluster(data)
    # Print the clusters
    printclust(clust, labels=row_names)