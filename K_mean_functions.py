import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def euclidian_distance(point1: np.array = None, point2: np.array = None) -> float:
    """Compute the euclidian distance of 2 points in n-th dimension

    Args:
        point1 (np.array): First point. Defaults to None.
        point2 (np.array): Second point. Defaults to None.

    Returns:
        float: Euclidian distance
    """
    return np.linalg.norm(point1 - point2)


def generate_cluster(
    mean: tuple = (0,),
    spread: tuple = (0,),
    numb_points: int = 1,
    dim: int = 1,
) -> np.array:
    """Generate a random cluster of points based on the parameters

    Args:
        mean (tuple, optional): Mean of the cluster. Defaults to (0,).
        spread (tuple, optional): Spread of the cluster. Defaults to (0,).
        numb_points (int, optional): Number of point in the cluster. Defaults to 1.
        dim (int, optional): Dimension of the cluster. Defaults to 1.

    Returns:
        np.array: Cluster of points
    """

    points = np.zeros((dim, numb_points))

    for i in range(dim):
        points[i] = np.random.normal(mean[i], spread[i], numb_points)
    return np.transpose(points)


def K_means(
    points: np.array = None,
    nb_clust: int = 0,
    max_it: int = np.inf,
    stop_val: float = 0.5,
    show_evolution: bool = False,
) -> tuple:
    """K-means method

    Args:
        points (np.array): The points to cluster. Defaults to None.
        nb_clust (int): Number of cluster also referd as K. Defaults to 0.
        max_it (int, optional): Max number of iteration. Defaults to np.inf.
        stop_val (float, optional): Min centers movement to stop the iterative evolution of the centers. Defaults to 0.5.
        show_evolution (bool, optional): use matplotlib to display the evolution of the identification (only in 2D and 3D). Defaults to False.

    Returns:
        tuple:
            - centers, a np.array with the coordonates of the center.
            - clusters_points a np.array with each cluster containing the values of the point of the cluster
            -  cluster_index a np.array with each cluster containing the index of each point from points
    """

    num_it = 0
    dim = len(points[0])

    # random generation of K centers in n-th dimension
    centers = np.zeros((nb_clust, dim), dtype=np.float64)
    distances = np.zeros((len(points), nb_clust))
    for i in range(nb_clust):
        centers[i] = points[np.random.randint(np.shape(points)[0])]

    new_centers = centers.copy()
    stop = False

    while (not stop) and num_it < max_it:
        num_it += 1
        clusters = [[] for i in range(nb_clust)]

        # computation of the distances

        for i in range(len(points)):
            for j in range(nb_clust):
                distances[i][j] = euclidian_distance(centers[j], points[i])

            # stockage de l'index des points du cluster dans le tableau cluster

            index = np.where(distances[i] == min(distances[i]))[0][0]
            clusters[index].append(i)

        # computation of the new centers

        stop_dist = 0
        for j in range(nb_clust):
            if np.shape(clusters[j])[0] != 0:
                new_centers[j] = (
                    sum([np.array(points[i], dtype=np.float64) for i in clusters[j]])
                    / np.shape(clusters[j])[0]
                )
                stop_dist += euclidian_distance(new_centers[j], centers[j])
            else:
                new_centers[j] = points[
                    np.where(
                        np.transpose(distances)[j] == min(np.transpose(distances)[j])
                    )[0][0]
                ]

        # update of the center or stop

        print(stop_dist, num_it)
        if stop_dist > stop_val:
            centers = new_centers.copy()
        else:
            stop = True

        # Plot if selected

        if show_evolution and (dim == 2):
            for cluster, center in zip(clusters, centers):
                plt.plot(
                    [points[i][0] for i in cluster],
                    [points[i][1] for i in cluster],
                    "x",
                )
                plt.plot(center[0], center[1], "o")

            plt.title(f"clusters, itteration {num_it}")
            # plt.savefig(f"image_D6_it_{num_it}.pdf")
            plt.show()

        if show_evolution and (dim == 3):
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            for i, (cluster, center) in enumerate(zip(clusters, centers)):
                if np.shape(cluster)[0] != 0:
                    print(np.shape(cluster)[0])
                    cluster_point = np.array([points[j] for j in cluster])
                    xs = np.transpose(cluster_point[::10])[0]
                    ys = np.transpose(cluster_point[::10])[1]
                    zs = np.transpose(cluster_point[::10])[2]
                    ax.scatter(
                        xs,
                        ys,
                        zs,
                        color=tuple(np.array(center, dtype=np.float32) / 255),
                        label=f"Cluster {i}",
                    )
                ax.scatter(center[0], center[1], center[2])

            print(centers)

            ax.set_xlabel("Red", color="r")
            ax.set_ylabel("Green", color="g")
            ax.set_zlabel("Blue", color="b")
            plt.legend()
            plt.title(f"clusters, itteration {num_it}")
            plt.show()
            # plt.savefig(f"image_D3_it_{num_it}.pdf")

    cluster_index = clusters
    # centers = new_centers.copy()
    clusters_points = [[points[j] for j in cluster_index[i]] for i in range(nb_clust)]

    return centers, clusters_points, cluster_index


def K_means_image(
    image_array: np.array = None,
    nb_clust: int = 0,
    max_it: int = np.inf,
    stop_val: float = 0.5,
    show_evolution: bool = False,
) -> None:

    # image_array = imageio.v2.imread(image_path)

    points, width = array_to_vect(image_array=image_array)

    centers, clusters_points, cluster_index = K_means(
        points=points,
        nb_clust=nb_clust,
        max_it=max_it,
        stop_val=stop_val,
        show_evolution=show_evolution,
    )

    for j in range(nb_clust):
        for i in range(np.shape(cluster_index[j])[0]):
            points[cluster_index[j][i]] = np.array(centers[j], dtype=np.uint8)
    image_array = vect_to_array(image_vector=points, width=width)
    cv.imwrite("k_means_segmented.png", image_array)
    return image_array,centers  


def array_to_vect(
    image_array: np.array = None,
) -> tuple:
    """Convert a image array into an image vector

    Args:
        image_array (np.array): image array. Defaults to None.

    Returns:
        tuple: (image_vector, width)
    """
    width = np.shape(image_array)[1]
    image_vector = np.zeros(
        (np.shape(image_array)[0] * np.shape(image_array)[1], 3), dtype=np.uint8
    )

    for i in range(np.shape(image_vector)[0]):
        image_vector[i] = image_array[i // width][i % width]

    return (image_vector, width)


def vect_to_array(image_vector: np.array = None, width: int = None) -> np.array:
    """Convert a vector image into an array image

    Args:
        image_vector (np.array): image vector. Defaults to None.
        width (int): image width. Defaults to None.

    Returns:
        np.array: image array
    """
    image_array = np.zeros(
        (np.shape(image_vector)[0] // width, width, 3), dtype=np.uint8
    )

    for i in range(np.shape(image_vector)[0]):
        image_array[i // width][i % width] = image_vector[i]

    return image_array
