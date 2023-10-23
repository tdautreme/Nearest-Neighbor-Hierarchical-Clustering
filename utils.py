import colorsys
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import time
import os

def get_datasets(n_samples = 1500, random_seed = None):
    if random_seed == None:
        random_seed = int(time.time())
    np.random.seed(random_seed)
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                        noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
    return {"noisy_circles": (noisy_circles[0], 2), "noisy_moons": (noisy_moons[0], 2), "varied": (varied[0], 3), "aniso": (aniso[0], 3), "blobs": (blobs[0], 3), "no_structure": (no_structure[0], 3)}
    
def generate_colors(nbr):
    N = nbr
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return list(RGB_tuples)

def cluster_screenshot(x, y, color_map, path="output.png", show=True):
    if "/" in path:
        path_split = path.split("/")
        path_split.pop()
        path_folder = "/".join(path_split)
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

    color_list = []
    for class_id in y:
        color_list.append(color_map[class_id])        
    plt.scatter(x[:,0], x[:,1], color = color_list)
    if show:
        plt.show()
    plt.savefig(path)
    plt.close()