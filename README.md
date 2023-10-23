# nearest-neighbor-hierarchical-clustering
This algorithm is an Agglomerative Hierarchical Clustering method that relies on the concept of nearest neighbors.<br>
Initially, every data point is assigned a unique label.<br>
Then, as long as there are more unique labels than a predefined number of clusters (denoted as 'n_clusters'), each data point identifies its nearest neighbor with a different label and adopts that label.<br>
<br>
However, it's important to note that this algorithm has a significant performance issue due to the usage of NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric=custom_distance) with a custom distance metric.<br>
In the custom distance method, the algorithm checks the label IDs in the x and y coordinates.<br>
If the labels are different, it returns an infinite distance to prevent those points from merging together.<br>
This can result in increased computation time and complexity.<br>

![noisy_circles](https://github.com/tdautreme/Nearest-Neighbor-Hierarchical-Clustering/blob/main/outputs/noisy_circles.png?raw=true)
![noisy_moons](https://github.com/tdautreme/Nearest-Neighbor-Hierarchical-Clustering/blob/main/outputs/noisy_moons.png?raw=true)
![varied](https://github.com/tdautreme/Nearest-Neighbor-Hierarchical-Clustering/blob/main/outputs/varied.png?raw=true)
![aniso](https://github.com/tdautreme/Nearest-Neighbor-Hierarchical-Clustering/blob/main/outputs/aniso.png?raw=true)
![blobs](https://github.com/tdautreme/Nearest-Neighbor-Hierarchical-Clustering/blob/main/outputs/blobs.png?raw=true)
![no_structure](https://github.com/tdautreme/Nearest-Neighbor-Hierarchical-Clustering/blob/main/outputs/no_structure.png?raw=true)
