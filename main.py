from utils import get_datasets, cluster_screenshot, generate_colors

# NNHC base class can be optimized (see Optimisation ideas below)

# less accurate but fast
from nnhc_centroids import NNHCCentroids # fast
from nnhc_full_nn_splitted_centroids import NNHCFullNNSplittedCentroids # less fast but more accurate (can be optimized)

# most accurate but slow
from nnhc_full_nn import NNHCFullNN # very slow (do not use)
from nnhc_full_nn_splitted import NNHCFullNNSplitted # less slow (can be optimized)

def get_nnhc(n_clusters):
    # return NNHCCentroids(n_clusters=n_clusters)
    # return NNHCFullNNSplittedCentroids(n_clusters=n_clusters)
    return NNHCFullNN(n_clusters=n_clusters)
    # return NNHCFullNNSplitted(n_clusters=n_clusters)

if __name__ == "__main__":
    datasets = get_datasets(n_samples=10000, random_seed=42)
    for dataset_name, (x, clusters_count) in datasets.items():
        # if dataset_name != "aniso":
            # continue
        print(f"\nClustering {dataset_name} with {clusters_count} clusters")
        nnhc = get_nnhc(n_clusters=clusters_count)
        nnhc.fit(x=x)
        color_map = generate_colors(len(nnhc.labels))
        cluster_screenshot(x, nnhc.y, path=f"outputs/{dataset_name}.png", color_map=color_map, show=False, print_numbers=False)
        # cluster_screenshot(nnhc.centroids, nnhc.labels, path=f"outputs/{dataset_name}_centroids.png", color_map=color_map, show=False, print_numbers=True)
        break
        
'''
    Optimisation ideas:
    - assign_label: do not assign labels, make a labels propagation main array that contain all references, assign it at the end
    - nnhc_centroids: do not recompute np.mean, track the sum and the count of each label, and compute the mean at the end 

    Todo:
    - create NearestNeighbor wrapper method in nnhc to timeit it
    - optimisations ideas -> make it    
'''