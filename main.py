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

class NNHC:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, x):
        x = np.array(x)
        # add index in the last column
        x = np.concatenate((x, np.arange(len(x)).reshape(-1, 1)), axis=1)
        # add label in the last column
        x = np.concatenate((x, np.arange(len(x)).reshape(-1, 1)), axis=1)
        
        while True:
            # get all differents labels without duplicates in x
            '''
                Here is the problem, check at custom_distance method
            '''
            x_without_ids = np.delete(x, -2, axis=1)
            checkpoint_time = time.time()
            neigh = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric=custom_distance).fit(x_without_ids)
            distances, indices = neigh.kneighbors(x_without_ids, return_distance=True)
            print("NearestNeighbors time: ", time.time() - checkpoint_time)

            indices = indices.reshape(-1)
            distances = distances.reshape(-1)
            neigh_original_indices = np.arange(len(distances))

            all_differents_labels = np.unique(x[:, -1])

            # create a label propagation array -> [[source_label, destination_label], ...]
            label_propagation_array = []

            # foreach diffrent label, get the point that have the minimum distance with another point
            for label in all_differents_labels:
                # get points that have the label
                points_with_label = x[x[:, -1] == label]
                # get their ids
                ids = points_with_label[:, -2]
                ids_np_int = np.array(ids, dtype=np.int32)

                # filter distances using ids_np_int
                filtered_distances = distances[ids_np_int]
                # find the index of the smallest distance in the filtered array
                min_distance_index = np.argmin(filtered_distances)
                # For getting the index in the original array, use original_indices
                min_distance_original_index = neigh_original_indices[ids_np_int[min_distance_index]]
            
                # Finaly, we have the value we wants. The point that have a different label and the minimum distance from a point of our label
                best_target_label = x[indices[min_distance_original_index], -1]
                # We can also get the distance from this target and the source
                best_distance = distances[min_distance_original_index]
    
                # add the result to the label propagation array
                label_propagation_array.append([label, best_target_label, best_distance])

            # reduce the label propagation array to the best label propagation
            label_propagation_array = np.array(label_propagation_array)
            propagation_order_array = process_propagation(label_propagation_array[:, :-1])

            # Add distance to propagation_order_array from label_propagation_array where [:, 0] are the same in each array 
            source_label_to_distance = {label: distance for label, _, distance in label_propagation_array}
            # add column for distance
            propagation_order_array = np.concatenate((propagation_order_array, np.zeros((len(propagation_order_array), 1))), axis=1)
            propagation_order_array[:, 2] = [source_label_to_distance[label] for label in propagation_order_array[:, 0]]
            # sort by distance
            propagation_order_array = propagation_order_array[propagation_order_array[:, 2].argsort()]
            # add offset to propagation_order_array
            propagation_order_array[:, 1] += np.max(x[:, -1]) + 1

            current_label_count = len(all_differents_labels)
            new_diffrent_labels = []
            # foreach label propagation, update the label in x
            end = False
            for source_label, target_label, _ in propagation_order_array:
                x[x[:, -1] == source_label, -1] = target_label
                current_label_count -= 1
                if target_label not in new_diffrent_labels:
                    new_diffrent_labels.append(target_label)

                # If we have enough labels, stop
                if current_label_count + len(new_diffrent_labels) == self.n_clusters:
                    end = True
                    break
            if end:
                break
        
        y = x[:, -1]
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
        print("")
        