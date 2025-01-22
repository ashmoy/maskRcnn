import os
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
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
        # Extraire le nom de l'image sans l'extension
        image_name = os.path.basename(image_path).replace(".nii", "").replace(".gz", "")

        # Rechercher les lignes correspondantes dans le fichier CSV (pour _L et _R)
        rows = labels_csv[labels_csv['Name'].str.startswith(image_name)]
        if rows.empty:
            print(f"Nom de l'image {image_name} non trouvé dans le fichier CSV. Passer à l'image suivante.")
            return

        # Obtenir les labels gauche et droite du fichier CSV
        label_L = rows['Label_L'].values
        label_R = rows['Label_R'].values

        # Lire l'image NIfTI
        nii_img = nib.load(image_path)
        img_data = nii_img.get_fdata()

        # Remplacer les labels dans l'image
        unique_labels = np.unique(img_data)
        print(f"Labels uniques dans l'image {image_name}: {unique_labels}",'taille',len(unique_labels))
        print('label_L',label_L)
        print('taille label_L',(label_L.size))


        # Si l'image contient seulement le label de gauche (1), le remplacer par Label_L
        if len(unique_labels) == 2 and 1 in unique_labels and label_R.size > 0:
            print('test1')
            img_data[img_data == 1] = label_R[0]  # Appliquer le mapping du label gauche
            print('test1')
            print(f"Image {image_name}: Remplacement du label gauche par {label_R[0]}")

        # Si l'image contient seulement le label de droite (2), le remplacer par Label_R
        elif len(unique_labels) == 2 and 2 in unique_labels and label_L.size > 0:
            print('test2')
            img_data[img_data == 2] = label_L[0] # Appliquer le mapping du label droit
            print('test2')
            print(f"Image {image_name}: Remplacement du label droit par {label_L[0]}")

        # Si l'image contient les deux labels, remplacer chacun par les labels correspondants
        elif len(unique_labels) == 3 and (2 in unique_labels and 1 in unique_labels):
            print('test3')
            if label_L.size > 0:
                print('test4')
                img_data[img_data == 2] = label_L [0] # Appliquer le mapping du label droit
            if label_R.size > 0:
                print('test5')
                img_data[img_data == 1] = label_R[0]  # Appliquer le mapping du label gauche
            print(f"Image {image_name}: Remplacement du label gauche par {label_L[0]} et du label droit par {label_R[0]}")

        # Sinon, ne pas toucher à l'image
        else:
            print(f"Image {image_name}: Aucun label ou label inconnu trouvé, pas de modification.")
            return

        # Sauvegarder l'image modifiée dans le dossier de sortie
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        labeled_img = nib.Nifti1Image(img_data, nii_img.affine)
        nib.save(labeled_img, output_path)
        print(f"Image sauvegardée: {output_path}")
    
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {image_name}: {e}")

def process_folder(input_folder, csv_file, output_folder):
    try:
        # Lire le fichier CSV
        labels_csv = pd.read_csv(csv_file)

        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_folder, exist_ok=True)

        # Parcourir toutes les images NIfTI du dossier d'entrée
        for image_file in os.listdir(input_folder):
            if image_file.endswith(".nii.gz") or image_file.endswith(".nii"):  # Assurez-vous que seuls les fichiers NIfTI sont traités
                image_path = os.path.join(input_folder, image_file)
                process_image(image_path, labels_csv, output_folder)

    except Exception as e:
        print(f"Erreur lors du traitement du dossier: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remplacer les labels des fichiers NIfTI segmentés par les valeurs du fichier CSV.")
    parser.add_argument("--input_folder", required=True, help="Dossier contenant les fichiers NIfTI segmentés (.nii.gz).")
    parser.add_argument("--csv_file", required=True, help="Fichier CSV contenant les labels pour chaque image.")
    parser.add_argument("--output_folder", required=True, help="Dossier où sauvegarder les fichiers NIfTI modifiés.")
    
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.csv_file, args.output_folder)
