from nnhc import NNHC, get_labels, reduce_label_propagation
from assign_labels import assign_labels_union_find
import numpy as np
from utils import timeit
from sklearn.neighbors import NearestNeighbors
import time
import numba as nb

@nb.njit(nopython=True)
def custom_distance(x, y):
    '''
        /!\ this methods take to much time to compute, it's the last thing to optimize to make a good clusturing algorithm
    '''
    # if same label, distance is infinite because we don't want to merge points with same label
    if x[-1] == y[-1]:
        return np.inf
    return np.linalg.norm(x[:-1] - y[:-1])

    '''
        This is fast but we can't compare labels...
    '''
    # return np.linalg.norm(x - y)

@timeit
def get_label_propagation_array_full_nn(x, y):
    x_with_labels = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    checkpoint_time = time.time()
    neigh = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric=custom_distance).fit(x_with_labels)
    distances, indices = neigh.kneighbors(x_with_labels, return_distance=True)
    print("NearestNeighbors time: ", time.time() - checkpoint_time)

    indices = indices.reshape(-1)
    distances = distances.reshape(-1)
    neigh_original_indices = np.arange(len(distances))

    labels = get_labels(y)

    # create a label propagation array -> [[source_label, destination_label], ...]
    label_propagation_array = []
    label_propagation_distances = []

    # foreach diffrent label, get the point that have the minimum distance with another point
    for label in labels:
        # get index of points that have the label
        point_indexes = np.where(y == label)[0]

        # filter distances using ids_np_int
        filtered_distances = distances[point_indexes]
        # find the index of the smallest distance in the filtered array
        min_distance_index = np.argmin(filtered_distances)
        # For getting the index in the original array, use original_indices
        min_distance_original_index = neigh_original_indices[point_indexes[min_distance_index]]
    
        # Finaly, we have the value we wants. The point that have a different label and the minimum distance from a point of our label
        best_target_label = y[indices[min_distance_original_index]]
        # We can also get the distance from this target and the source
        best_distance = distances[min_distance_original_index]

        # add the result to the label propagation array
        label_propagation_array.append([label, best_target_label])
        label_propagation_distances.append(best_distance)
    label_propagation_array = np.array(label_propagation_array)
    label_propagation_distances = np.array(label_propagation_distances)
    return reduce_label_propagation(label_propagation_array, distances=label_propagation_distances)

class NNHCFullNN(NNHC):
    def _get_label_propagation_array(self, x, y):
        return get_label_propagation_array_full_nn(x, y)
    
    def _assign_labels(self, label_propagation_array, y, labels):
        return assign_labels_union_find(label_propagation_array, y, labels, min_label_count=self.n_clusters)