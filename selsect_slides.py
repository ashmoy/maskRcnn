import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def save_slice_as_png(image, slice_idx, output_folder, file_name):
    """Sauvegarde une slice spécifique d'une image au format PNG."""
    if slice_idx >= image.shape[2]:
        print(f"Erreur: index {slice_idx} dépasse la taille de l'axe des slices ({image.shape[2]}) pour {file_name}.")
        return

    slice_img = image[:, :, slice_idx]
    output_path = os.path.join(output_folder, f"{file_name}_slice_{slice_idx}.png")
    plt.imsave(output_path, slice_img, cmap='gray')
    print(f"Saved {output_path}")

def find_best_slices(seg_data, num_slices=5, threshold=0.2, min_distance=10):
    """
    Cherche les 5 meilleures slices, en commençant par celle avec le plus de pixels non nuls,
    puis en ajoutant des slices ayant au moins 20% de pixels non nuls par rapport à la meilleure.
    S'assure qu'il y a une distance minimale entre les slices sélectionnées et ignore les slices sans composant non nul.
    """
    non_zero_counts = np.array([np.count_nonzero(seg_data[:, :, i]) for i in range(seg_data.shape[2])])

    # Exclure les slices avec 0 pixel non nul
    valid_slices = np.where(non_zero_counts > 0)[0]

    if len(valid_slices) == 0:
        print("Aucune slice avec des composants non nuls n'a été trouvée.")
        return []

    # Trouver l'index de la meilleure slice (celle avec le plus de pixels non nuls)
    best_slice_idx = valid_slices[np.argmax(non_zero_counts[valid_slices])]
    best_slice_count = non_zero_counts[best_slice_idx]

    # Trouver les autres slices avec au moins 20% de pixels non nuls par rapport à la meilleure
    valid_slices = valid_slices[np.where(non_zero_counts[valid_slices] >= threshold * best_slice_count)]

    # Filtrer les slices pour s'assurer qu'elles ne sont pas trop proches
    selected_slices = [best_slice_idx]  # Ajouter la meilleure slice
    for slice_idx in valid_slices:
        if all(abs(slice_idx - s) >= min_distance for s in selected_slices):  # Vérifier la distance
            selected_slices.append(slice_idx)
        if len(selected_slices) >= num_slices:
            break

    # Si on a moins de 5 slices, on complète avec les meilleures disponibles
    if len(selected_slices) < num_slices:
        sorted_indices = np.argsort(non_zero_counts[valid_slices])[::-1]  # Trier par nombre décroissant de pixels non nuls
        for slice_idx in valid_slices[sorted_indices]:
            if slice_idx not in selected_slices and all(abs(slice_idx - s) >= min_distance for s in selected_slices):
                selected_slices.append(slice_idx)
            if len(selected_slices) >= num_slices:
                break

    return selected_slices[:num_slices]

def extract_best_slices(scan_folder, seg_folder, output_folder):
    # Créer des sous-dossiers pour les scans et les segmentations
    scan_output_folder = os.path.join(output_folder, "scans")
    seg_output_folder = os.path.join(output_folder, "segmentations")
    os.makedirs(scan_output_folder, exist_ok=True)
    os.makedirs(seg_output_folder, exist_ok=True)

    # Parcourir les fichiers de scan dans le dossier d'entrée
    for scan_file in os.listdir(scan_folder):
        if scan_file.endswith(".nii.gz"):
            scan_path = os.path.join(scan_folder, scan_file)
            scan_data = nib.load(scan_path).get_fdata()

            # Trouver le fichier de segmentation correspondant
            seg_file = scan_file.replace("_scan.nii.gz", "_SEG.nii.gz")
            seg_path = os.path.join(seg_folder, seg_file)
            
            if os.path.exists(seg_path):
                seg_data = nib.load(seg_path).get_fdata()

                # Vérifier que les dimensions correspondent
                if scan_data.shape != seg_data.shape:
                    print(f"Erreur: Les dimensions du scan {scan_file} et de la segmentation {seg_file} ne correspondent pas.")
                    continue

                # Trouver les 5 meilleures slices avec la contrainte de distance et en évitant les slices avec 0 composant non nul
                best_slices = find_best_slices(seg_data, num_slices=15, threshold=0.2, min_distance=2)

                if best_slices:
                    # Sauvegarder les slices des scans et des segmentations
                    for slice_idx in best_slices:
                        save_slice_as_png(scan_data, slice_idx, scan_output_folder, scan_file.replace('.nii.gz', ''))
                        save_slice_as_png(seg_data, slice_idx, seg_output_folder, seg_file.replace('.nii.gz', ''))
                else:
                    print(f"Aucune slice valide trouvée pour {scan_file}")
            else:
                print(f"Fichier de segmentation correspondant pour {scan_file} introuvable.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraire les 5 meilleures slices à partir des scans et des segmentations correspondantes.")
    parser.add_argument("--scan_folder", help="Dossier contenant les fichiers de scan (.nii.gz)")
    parser.add_argument("--seg_folder", help="Dossier contenant les fichiers de segmentation (_SEG.nii.gz)")
    parser.add_argument("--output_folder", help="Dossier de sortie pour sauvegarder les slices au format PNG")

    args = parser.parse_args()
    
    # Lancer l'extraction des meilleures slices
    extract_best_slices(args.scan_folder, args.seg_folder, args.output_folder)
