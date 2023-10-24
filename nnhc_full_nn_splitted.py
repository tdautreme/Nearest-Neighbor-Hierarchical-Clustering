from nnhc import NNHC, get_labels, reduce_label_propagation
from assign_labels import assign_labels_union_find
import numpy as np
from utils import timeit
from sklearn.neighbors import NearestNeighbors
import time

@timeit
def get_label_propagation_array_full_nn_splitted(x, y):
    checkpoint_time = time.time()
    labels = get_labels(y)
    label_propagation_array = []
    indexes = np.arange(len(y))
    label_propagation_distances = []
    for label in labels:
        # get index of points that have the label
        mask = y == label
        x_label = x[mask]
        x_not_label = x[~mask]
        x_not_label_indexes = indexes[~mask]
        neigh = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(x_not_label)
        distances, indices = neigh.kneighbors(x_label, return_distance=True)
        distances = distances.reshape(-1)
        indices = indices.reshape(-1)

        min_distance_index = np.argmin(distances)
        x_not_label_index = x_not_label_indexes[indices[min_distance_index]]
    
        # Finaly, we have the value we wants. The point that have a different label and the minimum distance from a point of our label
        best_target_label = y[x_not_label_index]
        # We can also get the distance from this target and the source
        best_distance = distances[min_distance_index]

        # add the result to the label propagation array
        label_propagation_array.append([label, best_target_label])
        label_propagation_distances.append(best_distance)
    print("NearestNeighbors time: ", time.time() - checkpoint_time)
    label_propagation_array = np.array(label_propagation_array)
    label_propagation_distances = np.array(label_propagation_distances)
    return reduce_label_propagation(label_propagation_array, distances=label_propagation_distances)


class NNHCFullNNSplitted(NNHC):
    def _get_label_propagation_array(self, x, y):
        return get_label_propagation_array_full_nn_splitted(x, y)
    
    def _assign_labels(self, label_propagation_array, y, labels):
        return assign_labels_union_find(label_propagation_array, y, labels, min_label_count=self.n_clusters)