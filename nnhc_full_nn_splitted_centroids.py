from nnhc import NNHC, get_labels, reduce_label_propagation
from assign_labels import assign_labels_union_find
import numpy as np
from utils import timeit

@timeit
def get_label_propagation_array_nn_splitted_centroid(x, y):
    # reduce all x to the centroid of each label
    labels = get_labels(y)
    labels_count = labels.shape[0]
    centroids = []
    for label in labels:
        mask = y == label
        x_label = x[mask]
        centroid = np.mean(x_label, axis=0)
        centroids.append(centroid)
    label_propagation_array = []
    label_propagation_distances = []
    for i in range(labels_count):
        label_1 = labels[i]
        best_target_label = None
        best_distance = np.inf
        for j in range(labels_count):
            label_2 = labels[j]
            if label_1 == label_2:
                continue
            centroid_1 = centroids[i]
            centroid_2 = centroids[j]
            nearest_l1_point_from_l2_centroid = x[y == label_1][np.argmin(np.linalg.norm(x[y == label_1] - centroid_2, axis=1))]
            nearest_l2_point_from_l1_centroid = x[y == label_2][np.argmin(np.linalg.norm(x[y == label_2] - centroid_1, axis=1))]
            nearest_l1_point_from_nearest_l2_point_from_l1_centroid = x[y == label_1][np.argmin(np.linalg.norm(x[y == label_1] - nearest_l2_point_from_l1_centroid, axis=1))]
            nearest_l2_point_from_nearest_l1_point_from_l2_centroid = x[y == label_2][np.argmin(np.linalg.norm(x[y == label_2] - nearest_l1_point_from_l2_centroid, axis=1))]

            distance_1 = np.linalg.norm(nearest_l1_point_from_l2_centroid - nearest_l2_point_from_nearest_l1_point_from_l2_centroid)
            distance_2 = np.linalg.norm(nearest_l2_point_from_l1_centroid - nearest_l1_point_from_nearest_l2_point_from_l1_centroid)
            # get the best distance
            if distance_1 < best_distance:
                best_distance = distance_1
                best_target_label = label_2
            if distance_2 < best_distance:
                best_distance = distance_2
                best_target_label = label_2
        label_propagation_array.append([label_1, best_target_label])
        label_propagation_distances.append(best_distance)
    label_propagation_array = np.array(label_propagation_array)
    label_propagation_distances = np.array(label_propagation_distances)
    return reduce_label_propagation(label_propagation_array, label_propagation_distances)

class NNHCFullNNSplittedCentroids(NNHC):
    def _get_label_propagation_array(self, x, y):
        return get_label_propagation_array_nn_splitted_centroid(x, y)
    
    def _assign_labels(self, label_propagation_array, y, labels):
        return assign_labels_union_find(label_propagation_array, y, labels, min_label_count=self.n_clusters)