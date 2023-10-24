import time
import numpy as np
from utils import get_datasets, cluster_screenshot, generate_colors
from union_find import process_propagation
from sklearn.neighbors import NearestNeighbors
import numba as nb

'''
    Optimisation ideas:
        instead of making nearest neighbors with all points with a bad custom_distance methods,
        we can make unique_label_count nearest neighbors with all point from the same label with all other points

        make a fast version of NNHC wnich use the centroid of each label group as a point to compare with other centroid
'''

def reduce_label_propagation(label_propagation_array, distances=None):
    # remove distances for process if needed
    propagation_order_array = process_propagation(label_propagation_array)
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

def get_label_propagation_array_full_nn(x, labels):
    x_with_labels = np.concatenate((x, labels.reshape(-1, 1)), axis=1)
    checkpoint_time = time.time()
    neigh = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric=custom_distance).fit(x_with_labels)
    distances, indices = neigh.kneighbors(x_with_labels, return_distance=True)
    print("NearestNeighbors time: ", time.time() - checkpoint_time)

    indices = indices.reshape(-1)
    distances = distances.reshape(-1)
    neigh_original_indices = np.arange(len(distances))

    all_differents_labels = np.unique(labels)

    # create a label propagation array -> [[source_label, destination_label], ...]
    label_propagation_array = []
    label_propagation_distances = []

    # foreach diffrent label, get the point that have the minimum distance with another point
    for label in all_differents_labels:
        # get index of points that have the label
        point_indexes = np.where(labels == label)[0]

        # filter distances using ids_np_int
        filtered_distances = distances[point_indexes]
        # find the index of the smallest distance in the filtered array
        min_distance_index = np.argmin(filtered_distances)
        # For getting the index in the original array, use original_indices
        min_distance_original_index = neigh_original_indices[point_indexes[min_distance_index]]
    
        # Finaly, we have the value we wants. The point that have a different label and the minimum distance from a point of our label
        best_target_label = labels[indices[min_distance_original_index]]
        # We can also get the distance from this target and the source
        best_distance = distances[min_distance_original_index]

        # add the result to the label propagation array
        label_propagation_array.append([label, best_target_label])
        label_propagation_distances.append(best_distance)
    label_propagation_array = np.array(label_propagation_array)
    label_propagation_distances = np.array(label_propagation_distances)
    return reduce_label_propagation(label_propagation_array, distances=label_propagation_distances)

def get_label_propagation_unique_labels(x, labels, sort_distances=False):
    checkpoint_time = time.time()
    neigh = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(x)

    if sort_distances:
        distances, indices = neigh.kneighbors(x, return_distance=True)
    else:
        indices = neigh.kneighbors(x, return_distance=False)
    print("NearestNeighbors time: ", time.time() - checkpoint_time)
    # keep only the second column of indices and distances
    indices = indices[:, 1]
    label_propagation_array = labels[indices]
    label_propagation_array = np.concatenate((labels.reshape(-1, 1), label_propagation_array.reshape(-1, 1)), axis=1)

    if sort_distances:
        distances = distances[:, 1]
        return reduce_label_propagation(label_propagation_array, distances=distances)
    else:
        return reduce_label_propagation(label_propagation_array)

def get_label_propagation_array_centroid(x, labels):
    # reduce all x to the centroid of each label
    all_differents_labels = np.unique(labels)
    centroids = []
    for label in all_differents_labels:
        mask = labels == label
        x_label = x[mask]
        centroid = np.mean(x_label, axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    return get_label_propagation_unique_labels(centroids, all_differents_labels, sort_distances=True)

def get_label_propagation_array_nn_splitted_centroid(x, labels):
    # reduce all x to the centroid of each label
    all_differents_labels = np.unique(labels)
    centroids = []
    for label in all_differents_labels:
        mask = labels == label
        x_label = x[mask]
        centroid = np.mean(x_label, axis=0)
        centroids.append(centroid)
    label_propagation_array = []
    label_propagation_distances = []
    all_differents_labels_len = len(all_differents_labels)
    for i in range(all_differents_labels_len):
        label_1 = all_differents_labels[i]
        best_target_label = None
        best_distance = np.inf
        for j in range(all_differents_labels_len):
            label_2 = all_differents_labels[j]
            if label_1 == label_2:
                continue
            centroid_1 = centroids[i]
            centroid_2 = centroids[j]
            nearest_l1_point_from_l2_centroid = x[labels == label_1][np.argmin(np.linalg.norm(x[labels == label_1] - centroid_2, axis=1))]
            nearest_l2_point_from_l1_centroid = x[labels == label_2][np.argmin(np.linalg.norm(x[labels == label_2] - centroid_1, axis=1))]
            nearest_l1_point_from_nearest_l2_point_from_l1_centroid = x[labels == label_1][np.argmin(np.linalg.norm(x[labels == label_1] - nearest_l2_point_from_l1_centroid, axis=1))]
            nearest_l2_point_from_nearest_l1_point_from_l2_centroid = x[labels == label_2][np.argmin(np.linalg.norm(x[labels == label_2] - nearest_l1_point_from_l2_centroid, axis=1))]

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

def get_label_propagation_array_full_nn_splitted(x, labels):
    checkpoint_time = time.time()
    all_differents_labels = np.unique(labels)
    label_propagation_array = []
    indexes = np.arange(len(labels))
    label_propagation_distances = []
    for label in all_differents_labels:
        # get index of points that have the label
        mask = labels == label
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
        best_target_label = labels[x_not_label_index]
        # We can also get the distance from this target and the source
        best_distance = distances[min_distance_index]

        # add the result to the label propagation array
        label_propagation_array.append([label, best_target_label])
        label_propagation_distances.append(best_distance)
    print("NearestNeighbors time: ", time.time() - checkpoint_time)
    label_propagation_array = np.array(label_propagation_array)
    label_propagation_distances = np.array(label_propagation_distances)
    return reduce_label_propagation(label_propagation_array, distances=label_propagation_distances)


class NNHC:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, x):
        x = np.array(x)
        labels = np.arange(len(x));
        
        i = 0
        while True:
            all_differents_labels = np.unique(labels)
            print("labels count: ", len(all_differents_labels))

            # get the label_propagation_array
            if i == 0:
                label_propagation_array = get_label_propagation_unique_labels(x, labels)
            else:  
                # label_propagation_array = get_label_propagation_array_full_nn(x, labels) # BAD PERFORMANCE (but working)
                label_propagation_array = get_label_propagation_array_centroid(x, labels) # Not working well
                # label_propagation_array = get_label_propagation_array_full_nn_splitted(x, labels)  # Good
                # label_propagation_array = get_label_propagation_array_nn_splitted_centroid(x, labels) # To optimize with NearestNeighbors
    
            # add offset to propagation_order_array
            label_propagation_array[:, 1] += np.max(labels) + 1

            current_label_count = len(all_differents_labels)
            new_diffrent_labels = []
            # foreach label propagation, update the label in x
            end = False
            for source_label, target_label in label_propagation_array:
                # x[x[:, -1] == source_label, -1] = target_label
                labels[labels == source_label] = target_label
                current_label_count -= 1
                if target_label not in new_diffrent_labels:
                    new_diffrent_labels.append(target_label)

                # If we have enough labels, stop
                if current_label_count + len(new_diffrent_labels) == self.n_clusters:
                    end = True
                    break
            if end:
                break
            i += 1
        
        y = labels
        all_differents_labels = np.unique(y)
        # normalize label using indices (liek 0, 1, 2, 3, 4, ...)
        for i, label in enumerate(all_differents_labels):
            y[y == label] = i
        self.y = y.astype(np.int32)
        self.labels = all_differents_labels.astype(np.int32)
        return self

if __name__ == "__main__":
    datasets = get_datasets(n_samples=10000, random_seed=42)
    for dataset_name, (dataset_points, clusters_count) in datasets.items():
        print(f"Clustering {dataset_name} with {clusters_count} clusters")
        nnhc = NNHC(n_clusters=clusters_count)
        nnhc.fit(x=dataset_points)
        color_map = generate_colors(len(nnhc.labels))
        cluster_screenshot(dataset_points, nnhc.y, path=f"outputs/{dataset_name}.png", color_map=color_map, show=False)
        break