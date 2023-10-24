from nnhc import NNHC, get_labels, get_label_propagation_unique_labels
from assign_labels import assign_labels_union_find
import numpy as np
from utils import timeit

@timeit
def get_label_propagation_array_centroid(x, y):
    # reduce all x to the centroid of each label
    labels = get_labels(y)
    centroids = []
    for label in labels:
        mask = y == label
        x_label = x[mask]
        centroid = np.mean(x_label, axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    return get_label_propagation_unique_labels(centroids, labels, sort_distances=True)

class NNHCCentroids(NNHC):
    def _get_label_propagation_array(self, x, y):
        return get_label_propagation_array_centroid(x, y)
    
    def _assign_labels(self, label_propagation_array, y, labels):
        return assign_labels_union_find(label_propagation_array, y, labels, min_label_count=self.n_clusters)
    