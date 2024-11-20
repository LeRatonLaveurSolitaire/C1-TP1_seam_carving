import cv2 as cv
import numpy as np
import K_mean_functions as km

def calcul_energie_image(img,sobel=False):
    """
    Calcul l'énergie de chaque pixel avec la formule du laplacien\ (et ou la norme de sobel)\\
    """
    n,p=img.shape
    if sobel==False:
        laplacian = cv.Laplacian(img,cv.CV_64F)
        energie=np.abs(laplacian)
    else:
        sobel = cv.Sobel(img,cv.CV_64F,1,1,ksize=3)
        # Gestion des effets de bords en ajoutant une valeur arbitraire
        val=500
        sobel[0,:] = val
        sobel[-1,:] = val
        sobel[:,0] = val
        sobel[:,-1]=val
        # print(sobel[0])
        # assert False
        energie=np.abs(sobel)
    return energie

import numpy as np

def compute_optimal_seam_vertical(energy_map):
    """
    Computes the optimal seam in an image based on the given energy map using dynamic programming.

    Parameters:
        energy_map (np.ndarray): A 2D numpy array representing the energy of each pixel.

    Returns:
        seam (list): A list of row indices representing the optimal seam from top to bottom.
    """
    rows, cols = energy_map.shape
    
    # Création de la matrice de coût
    cost = np.zeros_like(energy_map)
    cost[0] = energy_map[0]  # on initialise la première ligne avec l'énergie de la première ligne

    # Calcul du coût pour chaque pixel
    for i in range(1, rows):
        for j in range(cols):
            # On prend le coût de l'énergie de la ligne précédente et on ajoute le minimum des trois pixels précédents
            min_cost = cost[i-1, j]  
            if j > 0:
                min_cost = min(min_cost, cost[i-1, j-1])  # Top-left diagonal
            if j < cols - 1:
                min_cost = min(min_cost, cost[i-1, j+1])  # Top-right diagonal
            
            # On ajoute le coût de l'énergie du pixel actuel
            cost[i, j] = energy_map[i, j] + min_cost

    # Recherche du seam optimal
    seam = []
    # On commence par le pixel de la dernière ligne avec le coût le plus faible
    min_index = np.argmin(cost[-1])
    seam.append(min_index)

    # On remonte la matrice de coût pour trouver le seam optimal
    for i in range(rows - 1, 0, -1):
        j = seam[-1]
        if j > 0 and cost[i-1, j-1] == cost[i, j] - energy_map[i, j]:
            min_index = j - 1
        elif j < cols - 1 and cost[i-1, j+1] == cost[i, j] - energy_map[i, j]:
            min_index = j + 1
        else:
            min_index = j
        
        seam.append(min_index)

    # On inverse le seam pour avoir les indices des pixels du haut vers le bas
    seam.reverse()
    # On convertit les indices en coordonnées (i, j)
    for i,j in enumerate(seam):
        seam[i] = [i,int(j)]

    return np.array(seam)

def compute_optimal_seam_horizontal(energy_map):
    """Cherche le seam optimal pour une image en transposant l'image et en utilisant la fonction compute_optimal_seam_vertical

    Args:
        energy_map (image): image de l'énergie

    Returns:
        seam: seam optimal
    """
    new_map = np.transpose(energy_map)
    seam = compute_optimal_seam_vertical(new_map)

    for i in range(np.shape(seam)[0]):
        seam[i] = seam[i,::-1]
    return seam
    
def mean_energy(energy_map,seam):
    """Calcul l'énergie moyenne d'un seam

    Args:
        energy_map (image): image de l'énergie
        seam (tableau): seam optimal

    Returns:
        float: énergie moyenne
    """
    energy_seam = [energy_map[i,j] for i,j in seam]
    return sum(energy_seam)/np.shape(seam)[0]

def calcul_energie_min_colonne(energy_map):
    """Calcul l'énergie minimale d'une colonne

    Args:
        energy_map (image): image de l'énergie

    Returns:
        energy,seam: énergie minimale et seam optimal
    """
    seam = compute_optimal_seam_vertical(energy_map)
    energy = mean_energy(energy_map,seam)
    return energy, seam

def calcul_energie_min_ligne(energy_map):
    """Calcul l'énergie minimale d'une ligne

    Args:
        energy_map (image): image de l'énergie

    Returns:
        energy,seam: énergie minimale et seam optimal
    """
    seam = compute_optimal_seam_horizontal(energy_map)
    energy = mean_energy(energy_map,seam)
    return energy, seam

def supprimer_colonne(img,Lpixel):
    """
    Supprime la colonne Lpixel de l'image img\\
    img de dimension 3\\
    Lpixel [x,y] à supprimer. Un par ligne\\
    Retourne l'image sans la colonne
    """
    taille=img.shape
    if len(taille)==2:
        n,p=img.shape
        new_img=np.zeros((n,p-1))
    else:
        n,p,q=img.shape
        new_img=np.zeros((n,p-1,q))
    # Pour chaque ligne, on supprime le pixel voulu
    for i in range(n):
        index=Lpixel[i]
        new_img[i]=np.delete(img[i],index[1],0)
    n,p,q=new_img.shape
    # print(n,p)
    # On pense à reconvertir dans le bon format pour l'affichage
    return np.array(new_img,dtype=np.uint8)

def supprimer_ligne(img,Lpixel):
    """
    Supprime la colonne Lpixel de l'image img\\
    img de dimension 3\\
    Lpixel [x,y] à supprimer. Un par colonne\\
    Retourne l'image sans la ligne
    """
    taille=img.shape
    if len(taille)==2:
        n,p=img.shape
        new_img=np.zeros((n-1,p))
    else:
        n,p,q=img.shape
        new_img=np.zeros((n-1,p,q))
    # Pour chaque colonne, on supprime le pixel voulu
    for i in range(p):
        index=Lpixel[i]
        new_img[:,i]=np.delete(img[:,i],index[0],0)
    n,p,q=new_img.shape
    # print(n,p)
    # On pense à reconvertir dans le bon format pour l'affichage
    return np.array(new_img,dtype=np.uint8)


def seam_carving(path_name='plage.jpg',taille_finale=[150,150],segmentation=False,affichage=False,sobel=False):
    """Réduit la taille d'une image en supprimant les colonnes et les lignes avec l'énergie la plus faible

    Args:
        path_name (str, optional): image à réduire. Defaults to 'plage.jpg'.
        taille_finale (list, optional): taille finale voulu. Defaults to [150,150].
        segmentation (bool, optional): on procède à la segmentation. Defaults to False.
        affichage (bool, optional): procède à l'affichage. Defaults to False.
        sobel (bool, optional): utilisation d'un filtre de sobel à la place d'un filtre laplacien pour calculer l'énergie. Defaults to False.
    """
    print('Traitement de l\'image {}, taille finale {}'.format(path_name,taille_finale))
    image_couleur = cv.imread(path_name)


    # On procède à la segmentation avec la méthode K-means si demandé
    if segmentation:
        # Nb de cluster
        nb_clusters=4
        print('Segmentation en {} cluster'.format(nb_clusters))
        image_segmenter,couleur=km.K_means_image(image_couleur,nb_clusters,20)
        cv.imwrite("{}_segmentation.jpg".format(path_name[:-4]), image_segmenter)
        # le noir (couleur la plus foncé) est la couleur à ne pas modifier pour l'image plage: à adapté pour chaque image
        # print(couleur)
        id=0
        valmin=255
        for i in range(nb_clusters):
            if np.sum(couleur[i])/3<valmin:
                valmin=couleur[i][0]
                id=i

    n,p,q=image_couleur.shape
    # Calcul du nombre d'itération
    iteration_ligne=n-taille_finale[0]
    iteration_colonne=p-taille_finale[1]    
    # On procède de façon itératif
    for i in range(iteration_ligne+iteration_colonne):
        # Conversion de l'image en niveau de gris et calcul de l'énergie
        image_gris = cv.cvtColor(image_couleur, cv.COLOR_BGR2GRAY)
        image_energie=calcul_energie_image(image_gris,sobel)    #False permet l'utilisation du filtre Laplacien, True Sobel

        # On met l'énergie maximale sur les pixels de la couleur qui correspond à notre choix de segmentation
        if segmentation:
            energie_max=np.max(image_energie)
            n,p=image_energie.shape
            for i in range(n):
                for j in range(p):
                    if abs(np.sum(image_segmenter[i][j])-np.sum(couleur[id]))<10:
                        image_energie[i][j]=energie_max

        # Calcul du flot minimale sur les lignes et les colonnes
        Colonne=calcul_energie_min_colonne(image_energie)
        Ligne=calcul_energie_min_ligne(image_energie)

        # On affiche les deux flots minimaux
        if affichage:
            image_aff=image_couleur.copy()
            for pixel in Colonne[1]:
                image_aff[pixel[0]][pixel[1]]=[0,0,255]
            for pixel in Ligne[1]:
                image_aff[pixel[0]][pixel[1]]=[0,0,255]
            cv.imshow('image', image_aff)
            cv.waitKey(10)


        # On choisit si on réduit sur les lignes ou les colonnes en prenant l'énergie moyenne par pixel minimale
        # On applique ensuite la réduction
        if (Colonne[0]<Ligne[0] and iteration_colonne>0) or iteration_ligne==0:
            iteration_colonne-=1
            Lpixel=Colonne[1]
            image_couleur = supprimer_colonne(image_couleur,Lpixel)
            if segmentation:
                image_segmenter = supprimer_colonne(image_segmenter,Lpixel)
        else:
            iteration_ligne-=1
            Lpixel=Ligne[1]
            image_couleur = supprimer_ligne(image_couleur,Lpixel)
            if segmentation:
                image_segmenter = supprimer_ligne(image_segmenter,Lpixel)

        
    # Affichage et enregistrement de l'image finale
    cv.imwrite("{}_resultat.jpg".format(path_name[:-4]), image_couleur)
    if affichage:
        cv.imshow('image', image_couleur)
        cv.waitKey(1000)
    # cv.destroyAllWindows()

if __name__ == "__main__":
    seam_carving('plage.jpg',[150,150],segmentation=False,affichage=True)
    seam_carving('plage.jpg',[150,150],segmentation=True,affichage=True)
    # seam_carving('oiseau.jpg',[800,800],segmentation=False,affichage=True,sobel=False)
    pass

print('fin')
    