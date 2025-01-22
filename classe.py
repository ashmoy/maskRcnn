import os
import argparse
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
import re

# Dictionnaire de conversion des labels du CSV vers les valeurs d'intensité finale
label_mapping = {
    3: 255,
    2: int(0.75 * 255),
    1: int(0.5 * 255),
    0: 0
}

def process_image(image_path, labels_csv, output_folder):
    try:
        # Extraire le nom de l'image avant le premier underscore pour correspondre à la colonne "Name" du fichier CSV
        image_name = os.path.basename(image_path)
        image_base = re.match(r"([^_]+)", image_name).group(1)

        # Rechercher la ligne correspondante dans le fichier CSV
        row = labels_csv[labels_csv['Name'].str.contains(image_base)]
        if row.empty:
            print(f"Nom de l'image {image_base} non trouvé dans le fichier CSV. Passer à l'image suivante.")
            return

        # Obtenir les labels gauche et droite du fichier CSV
        label_L = row['Label_L'].values[0]
        label_R = row['Label_R'].values[0]

        # Lire l'image
        img = imread(image_path)

        # Remplacer les labels dans l'image
        unique_labels = np.unique(img)

        # Si l'image contient seulement le label de gauche (255), le remplacer par Label_L
        if len(unique_labels) == 2 and 255 in unique_labels:  # en supposant que 0 est le fond
            img[img == 255] = label_mapping.get(label_R, 255)  # Appliquer le mapping du label gauche
            print(f"Image {image_name}: Remplacement du label gauche par {label_R}")

        # Si l'image contient seulement le label de droite (128), le remplacer par Label_R
        elif len(unique_labels) == 2 and 128 in unique_labels:
            img[img == 128] = label_mapping.get(label_L, 255)  # Appliquer le mapping du label droit
            print(f"Image {image_name}: Remplacement du label droit par {label_L}")

        # Si l'image contient les deux labels, remplacer chacun par les labels correspondants
        elif len(unique_labels) == 3 and (128 in unique_labels and 255 in unique_labels):
            img[img == 255] = label_mapping.get(label_R, 255)  # Appliquer le mapping du label gauche
            img[img == 128] = label_mapping.get(label_L, 255)  # Appliquer le mapping du label droit
            print(f"Image {image_name}: Remplacement du label gauche par {label_R} et du label droit par {label_L}")

        # Sinon, ne pas toucher à l'image
        else:
            print(f"Image {image_name}: Aucun label ou label inconnu trouvé, pas de modification.")
            return

        # Sauvegarder l'image modifiée dans le dossier de sortie
        output_path = os.path.join(output_folder, image_name)
        imsave(output_path, img)
        print(f"Image sauvegardée: {output_path}")
    
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {image_name}: {e}")

def process_folder(input_folder, csv_file, output_folder):
    try:
        # Lire le fichier CSV
        labels_csv = pd.read_csv(csv_file)

        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_folder, exist_ok=True)

        # Parcourir toutes les images du dossier d'entrée
        for image_file in os.listdir(input_folder):
            if image_file.endswith(".png"):  # Assurez-vous que seuls les fichiers PNG sont traités
                image_path = os.path.join(input_folder, image_file)
                process_image(image_path, labels_csv, output_folder)

    except Exception as e:
        print(f"Erreur lors du traitement du dossier: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remplacer les labels d'images segmentées par les valeurs du fichier CSV.")
    parser.add_argument("--input_folder", required=True, help="Dossier contenant les images segmentées (PNG).")
    parser.add_argument("--csv_file", required=True, help="Fichier CSV contenant les labels pour chaque image.")
    parser.add_argument("--output_folder", required=True, help="Dossier où sauvegarder les images modifiées.")
    
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.csv_file, args.output_folder)
