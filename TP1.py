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
        val=255
        sobel[0,:] = val
        sobel[-1,:] = val
        sobel[:,0] = val
        sobel[:,-1]
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
    # Get the dimensions of the energy map
    rows, cols = energy_map.shape
    
    # Create a cost matrix to store the minimum cost to reach each pixel
    cost = np.zeros_like(energy_map)
    cost[0] = energy_map[0]  # Initialize the first row of cost with the first row of energy

    # Fill in the cost matrix
    for i in range(1, rows):
        for j in range(cols):
            # Get the minimum cost from the previous row
            min_cost = cost[i-1, j]  # Directly above
            if j > 0:
                min_cost = min(min_cost, cost[i-1, j-1])  # Top-left diagonal
            if j < cols - 1:
                min_cost = min(min_cost, cost[i-1, j+1])  # Top-right diagonal
            
            # Update the cost for this pixel
            cost[i, j] = energy_map[i, j] + min_cost

    # Backtrack to find the optimal seam
    seam = []
    # Start from the last row and find the index of the minimum value in that row
    min_index = np.argmin(cost[-1])
    seam.append(min_index)

    for i in range(rows - 1, 0, -1):
        j = seam[-1]
        # Check the three possible positions from which we can come
        if j > 0 and cost[i-1, j-1] == cost[i, j] - energy_map[i, j]:
            min_index = j - 1
        elif j < cols - 1 and cost[i-1, j+1] == cost[i, j] - energy_map[i, j]:
            min_index = j + 1
        else:
            min_index = j
        
        seam.append(min_index)

    # Reverse the seam to get it from top to bottom
    seam.reverse()

    for i,j in enumerate(seam):
        seam[i] = [i,int(j)]

    return np.array(seam)
# Example usage:
# optimal_seam = compute_optimal_seam(energy_map)

def compute_optimal_seam_horizontal(energy_map):
    new_map = np.transpose(energy_map)
    seam = compute_optimal_seam_vertical(new_map)

    for i in range(np.shape(seam)[0]):
        seam[i] = seam[i,::-1]
    return seam
    
def mean_energy(energy_map,seam):
    energy_seam = [energy_map[i,j] for i,j in seam]
    return sum(energy_seam)/np.shape(seam)[0]

def calcul_energie_min_colonne(energy_map):
    seam = compute_optimal_seam_vertical(energy_map)
    energy = mean_energy(energy_map,seam)
    return energy, seam

def calcul_energie_min_ligne(energy_map):
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


def main():
    path_name = 'oiseau.jpg'
    image_couleur = cv.imread(path_name)
    n,p,q=image_couleur.shape
    # print(n,p)
    # assert False
    taille_finale=[800,800]
    # Calcul du nombre d'itération
    iteration_ligne=n-taille_finale[0]
    iteration_colonne=p-taille_finale[1]    
    # On procède de façon itératif
    for i in range(iteration_ligne+iteration_colonne):
        # Conversion de l'image en niveau de gris et calcul de l'énergie
        image_gris = cv.cvtColor(image_couleur, cv.COLOR_BGR2GRAY)
        image_energie=calcul_energie_image(image_gris,False)    #False permet l'utilisation du filtre Laplacien, True Sobel

        # Calcul du flot minimale sur les lignes et les colonnes
        Colonne=calcul_energie_min_colonne(image_energie)
        Ligne=calcul_energie_min_ligne(image_energie)

        # On affiche les deux flots minimaux
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
        else:
            iteration_ligne-=1
            Lpixel=Ligne[1]
            image_couleur = supprimer_ligne(image_couleur,Lpixel)

        
    # Affichage et enregistrement de l'image finale
    cv.imwrite("{}_resultat_sans_seg.jpg".format(path_name[:-4]), image_couleur)
    cv.imshow('image', image_couleur)

    cv.waitKey(1000)
    # cv.destroyAllWindows()


def main2():
    path_name = 'oiseau.jpg'
    image_couleur = cv.imread(path_name)

    image_segmenter,couleur=km.K_means_image(image_couleur,2,20)
    # le noir (couleur la plus foncé) est la couleur à ne pas modifier
    cv.imwrite("{}_segmentation.jpg".format(path_name[:-4]), image_segmenter)
    print(couleur)
    assert False
    id=0
    valmin=255
    for i in range(4):
        if np.sum(couleur[i])/3<valmin:
            valmin=couleur[i][0]
            id=i

    n,p,q=image_couleur.shape
    # print(n,p)
    # assert False
    taille_finale=[800,800]
    # Calcul du nombre d'itération
    iteration_ligne=n-taille_finale[0]
    iteration_colonne=p-taille_finale[1]    
    # On procède de façon itératif
    for i in range(iteration_ligne+iteration_colonne):
        # Conversion de l'image en niveau de gris et calcul de l'énergie
        image_gris = cv.cvtColor(image_couleur, cv.COLOR_BGR2GRAY)
        image_energie=calcul_energie_image(image_gris,False)    #False permet l'utilisation du filtre Laplacien, True Sobel

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
            image_segmenter = supprimer_colonne(image_segmenter,Lpixel)
        else:
            iteration_ligne-=1
            Lpixel=Ligne[1]
            image_couleur = supprimer_ligne(image_couleur,Lpixel)
            image_segmenter = supprimer_ligne(image_segmenter,Lpixel)

        
    # Affichage et enregistrement de l'image finale
    cv.imwrite("{}_resultat_avec_seg.jpg".format(path_name[:-4]), image_couleur)
    cv.imshow('image', image_couleur)
    cv.waitKey(1000)
    # cv.destroyAllWindows()

if __name__ == "__main__":
    print('Main1')
    # main()
    print('Main2')
    main2()


print('fin')
    