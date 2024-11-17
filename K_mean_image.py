from K_mean_functions import *

image_path = "montagne.jpg"

# Computation of the K-means on the image

K_means_image(
    image_path=image_path,
    nb_clust=3,
    max_it=50,
    stop_val=0.001,
    show_evolution=True,
)
