import time
import numpy as np
from utils import timeit
from union_find import process_propagation, union_find
from sklearn.neighbors import NearestNeighbors

'''
    Optimisation ideas:
        instead of making nearest neighbors with all points with a bad custom_distance methods,
        we can make unique_label_count nearest neighbors with all point from the same label with all other points

        make a fast version of NNHC wnich use the centroid of each label group as a point to compare with other centroid
'''

@timeit
def reduce_label_propagation(label_propagation_array, distances=None):
    # remove distances for process if needed
    # propagation_order_array = process_propagation(label_propagation_array)
    propagation_order_array = union_find(label_propagation_array)
    return propagation_order_array
    if distances is not None:
        # label_propagation_array[:, 0] and propagation_order_array[:, 0] are the keys to sert indices in the good order
        # label_propagation_array = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
        # propagation_order_array = [[0, 1], [2, 3], [4, 5], [1, 2], [3, 4]]
        # indices = [0, 3, 1, 4, 2]
        indices = np.searchsorted(label_propagation_array[:, 0], propagation_order_array[:, 0])
        # use indices to sort propagation_order_array
        propagation_order_array = propagation_order_array[indices]
        # use distances to sort propagation_order_array
        propagation_order_array = propagation_order_array[np.argsort(distances)]
    return propagation_order_array

@timeit
def get_label_propagation_unique_labels(x, y, sort_distances=False):
    checkpoint_time = time.time()
    neigh = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(x)

    if sort_distances:
        distances, indices = neigh.kneighbors(x, return_distance=True)
    else:
        indices = neigh.kneighbors(x, return_distance=False)
    print("NearestNeighbors time: ", time.time() - checkpoint_time)
    # keep only the second column of indices and distances
    indices = indices[:, 1]
    label_propagation_array = y[indices]
    label_propagation_array = np.concatenate((y.reshape(-1, 1), label_propagation_array.reshape(-1, 1)), axis=1)

    if sort_distances:
        distances = distances[:, 1]
        return reduce_label_propagation(label_propagation_array, distances=distances)
    else:
        return reduce_label_propagation(label_propagation_array)

# return y normalized and different labels
@timeit
def normalize_labels(y):
    labels = get_labels(y)
    # normalize label using indices (liek 0, 1, 2, 3, 4, ...)
    for i, label in enumerate(labels):
        y[y == label] = i
    y = y.astype(np.int32)
    return y, np.arange(len(labels))

@timeit
def get_labels(y):
    return np.unique(y)

class NNHC:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    @property
    def centroids(self):
        labels = self.labels
        x = self.x
        y = self.y
        centroids = []
        for label in labels:
            mask = y == label
            x_label = x[mask]
            centroid = np.mean(x_label, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        return centroids
    
    # to override
    def _get_label_propagation_array(self, x, y):
        pass

    # to verride
    def _assign_labels(self, label_propagation_array, y, labels):
        pass
    
    def _fit(self, x, y, labels):
        i = 0
        while True:
            print("\nlabels count: ", len(labels))
            if i == 0:
                label_propagation_array = get_label_propagation_unique_labels(x, y)
            else:
                label_propagation_array = self._get_label_propagation_array(x, y)

            y, labels = self._assign_labels(label_propagation_array, y, labels)
            labels_count = labels.shape[0]
            if labels_count <= self.n_clusters:
                break
            i += 1
        return y

    def fit(self, x):
        # prepare input
        x = np.array(x)
        self.x = x
        y = np.arange(len(x));
        labels = get_labels(y)
        # fit
        y = self._fit(x, y, labels)
        # prepare output
        y, labels = normalize_labels(y)
        self.y = y.astype(np.int32)
        self.labels = labels
        return self
