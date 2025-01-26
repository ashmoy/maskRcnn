import os
import nibabel as nib
import numpy as np

def filter_labels_nii_by_chunks(file_path, allowed_labels, chunk_size=64):
    try:
        img = nib.load(file_path)
        data = img.dataobj  # Utilisation de mmap pour un accès paresseux
        shape = data.shape

        # Préparer une image de sortie vide
        filtered_data = np.zeros(shape, dtype=np.uint8)

        # Itérer sur les sous-volumes
        for z in range(0, shape[2], chunk_size):
            for y in range(0, shape[1], chunk_size):
                for x in range(0, shape[0], chunk_size):
                    # Charger un sous-volume
                    x_end = min(x + chunk_size, shape[0])
                    y_end = min(y + chunk_size, shape[1])
                    z_end = min(z + chunk_size, shape[2])

                    chunk = data[x:x_end, y:y_end, z:z_end]
                    chunk = np.round(chunk)  # Arrondir les valeurs des labels
                    filtered_chunk = np.where(np.isin(chunk, allowed_labels), chunk, 0)
                    filtered_data[x:x_end, y:y_end, z:z_end] = filtered_chunk

        # Vérification des labels uniques après filtrage
        unique_labels_after = np.unique(filtered_data)
        print(f"Labels restants après filtrage : {unique_labels_after}")

        # Sauvegarder l'image filtrée
        filtered_img = nib.Nifti1Image(filtered_data, img.affine, img.header)
        output_file_path = file_path.replace('.nii', '_filtered.nii.gz')
        nib.save(filtered_img, output_file_path)

        print(f"Fichier traité avec succès : {output_file_path}")
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {file_path}: {e}")


# Parcourir un dossier et filtrer les fichiers de labels
def process_label_files(folder_path, allowed_labels):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nii'):  # Vérifie l'extension des fichiers
                file_path = os.path.join(root, file)
                filter_labels_nii_by_chunks(file_path, allowed_labels)

if __name__ == "__main__":
    folder_path = input("Entrez le chemin du dossier contenant les fichiers de labels : ")
    allowed_labels = [1.0, 2.0, 3.0]  # Les labels que vous voulez conserver

    if os.path.exists(folder_path):
        process_label_files(folder_path, allowed_labels)
    else:
        print("Le dossier spécifié n'existe pas.")
