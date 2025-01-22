import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

# Charger le fichier CSV
csv_path = "data.csv"
df = pd.read_csv(csv_path, sep=",")  # Assurez-vous que le séparateur est correct (ici tabulation)

# Dictionnaire pour les correspondances de positions
position_to_label = {
    "Buccal": 1,
    "Bicortical": 2,
    "Palatal": 3
}
# Afficher les colonnes présentes dans le CSV
print("Colonnes disponibles dans le fichier CSV :", list(df.columns))
def process_segmentations(input_folder, output_folder, df):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Parcourir tous les fichiers de segmentation
    for seg_file in tqdm(os.listdir(input_folder), desc="Processing segmentations"):
        if not seg_file.endswith(".nii.gz"):
            continue
        
        seg_path = os.path.join(input_folder, seg_file)
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()
        
        scan_id = os.path.splitext(os.path.splitext(seg_file)[0])[0]  # Exclure extensions .nii.gz
        
        # Extraire les lignes correspondantes dans le fichier CSV
        relevant_rows = df[df["Patient"] == scan_id]
        
        # Vérifier combien de canines sont présentes dans le fichier de segmentation
        unique_labels = np.unique(seg_data)
        unique_labels = [label for label in unique_labels if label in [6, 11]]  # Labels attendus
        print(f"scan_id: {scan_id}, unique_labels: {unique_labels}")
        
        for label in unique_labels:
            if label == 6:  # Canine droite
                canine_side = "R"
            elif label == 11:  # Canine gauche
                canine_side = "L"
            else:
                continue  # Ignorer les autres labels
            
            if len(unique_labels) == 1:  # Une seule canine
                position = relevant_rows["Position"].values
                print(f"scan_id: {scan_id}, canine_side: {canine_side}, position: {position}")
            else:  # Deux canines, regarder la colonne "Patient"
                canine_id = f"{scan_id}_{canine_side}"
                position = relevant_rows[relevant_rows["Canines"] == canine_id]["Position"].values
                print(f"scan_id: {scan_id}, canine_side: {canine_side}, position: {position}")
            
            # Traduire la position en nouveau label
            new_label = position_to_label.get(position[0], 0)
            
            # Mettre à jour les labels dans les données de segmentation
            seg_data[seg_data == label] = new_label
        
        # Sauvegarder le fichier de segmentation modifié
        output_path = os.path.join(output_folder, seg_file)
        new_seg_img = nib.Nifti1Image(seg_data, seg_img.affine, seg_img.header)
        nib.save(new_seg_img, output_path)

# Paramètres des dossiers
input_folder = "/home/luciacev/Desktop/enzo/maskrcnn/bad_seg"  # Dossier contenant les fichiers de segmentation
output_folder = "/home/luciacev/Desktop/enzo/maskrcnn/seg_with_classes"  # Dossier de sortie

# Exécuter le traitement
process_segmentations(input_folder, output_folder, df)
