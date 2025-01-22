import os
import csv
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.ops.boxes import masks_to_boxes
from torch import nn, optim
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
from engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


# ============================
# Utilisation du GPU si dispo
# ============================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


############################################################
# (A) EXTRACTION & SAUVEGARDE DES TRANCHES
############################################################
def extract_and_save_slices(
    scans_folder,
    masks_folder,
    temp_folder,
    csv_path,
    min_box_size=35
):
    """
    1) Parcourt les (scan, masque) en .nii dans scans_folder et masks_folder
    2) Pour chaque tranche 2D :
       - Vérifie bounding box >= min_box_size
       - Vérifie qu'il y a au moins un label > 0
       - Sauvegarde en .npy + écrit dans le CSV
    """
    os.makedirs(temp_folder, exist_ok=True)

    scan_files = sorted([f for f in os.listdir(scans_folder) if f.endswith('.nii')])
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.nii')])
    assert len(scan_files) == len(mask_files), "Mismatch entre scans et masques!"

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['slice_id', 'scan_npy', 'mask_npy'])
        slice_id = 0

        for scan_file, mask_file in tqdm(
            zip(scan_files, mask_files),
            total=len(scan_files),
            desc="Extraction & Sauvegarde des slices"
        ):
            scan_path = os.path.join(scans_folder, scan_file)
            mask_path = os.path.join(masks_folder, mask_file)

            try:
                # mmap pour éviter de tout charger en mémoire
                scan_img = nib.load(scan_path, mmap=True)
                mask_img = nib.load(mask_path, mmap=True)

                scan_proxy = scan_img.dataobj
                mask_proxy = mask_img.dataobj

                nb_slices_scan = scan_img.shape[2]
                nb_slices_mask = mask_img.shape[2]
                min_slices = min(nb_slices_scan, nb_slices_mask)
            except Exception as e:
                print(f"Erreur chargement {scan_file} ou {mask_file}: {e}")
                continue

            for i in range(min_slices):
                scan_slice = np.asanyarray(scan_proxy[..., i])
                mask_slice = np.asanyarray(mask_proxy[..., i])

                # Vérif bounding box
                mask_tensor = torch.from_numpy(mask_slice)
                y, x = torch.where(mask_tensor != 0)
                if len(y) == 0:
                    continue
                y_min, y_max = y.min(), y.max()
                x_min, x_max = x.min(), x.max()
                height = y_max - y_min
                width  = x_max - x_min
                if height < min_box_size or width < min_box_size:
                    continue

                # Vérif qu'il y a AU MOINS un label > 0
                unique_vals = torch.unique(mask_tensor)
                # On enlève la valeur 0, il reste éventuellement [1, 2, 3, ...]
                valid_vals = unique_vals[unique_vals > 0]
                if len(valid_vals) == 0:
                    # => masque n'a pas de classe d'objet > 0
                    continue

                # Normalisation
                min_val, max_val = scan_slice.min(), scan_slice.max()
                if (max_val - min_val) > 1e-8:
                    scan_slice = (scan_slice - min_val) / (max_val - min_val)
                else:
                    scan_slice = np.zeros_like(scan_slice)

                # Sauvegarde .npy
                scan_outfile = os.path.join(temp_folder, f"scan_{slice_id}.npy")
                mask_outfile = os.path.join(temp_folder, f"mask_{slice_id}.npy")
                np.save(scan_outfile, scan_slice)
                np.save(mask_outfile, mask_slice)

                # Écrit dans le CSV
                writer.writerow([slice_id, scan_outfile, mask_outfile])
                slice_id += 1

    print(f"Extraction terminée. Les tranches valides sont dans {temp_folder}")
    print(f"Métadonnées enregistrées dans : {csv_path}")


############################################################
# (B) DATASET
############################################################
class TempSlicesDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        """
        On suppose que le CSV ne contient QUE des tranches valides
        (i.e. bounding box >= min_box_size + au moins 1 label > 0).
        """
        self.transforms = transforms

        # Lire le CSV
        self.entries = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slice_id = int(row['slice_id'])
                scan_npy = row['scan_npy']
                mask_npy = row['mask_npy']
                self.entries.append((slice_id, scan_npy, mask_npy))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        slice_id, scan_path, mask_path = self.entries[idx]

        # Charger .npy
        scan_slice = np.load(scan_path)  # [H, W]
        mask_slice = np.load(mask_path)  # [H, W]

        scan_tensor = torch.from_numpy(scan_slice).unsqueeze(0).float()  # [1, H, W]
        mask_tensor = torch.from_numpy(mask_slice).long()                # [H, W]

        # Labels > 0 => objets
        labels = torch.unique(mask_tensor)
        labels = labels[labels != 0]  # On retire 0 (fond)
        # À ce stade, on SAVAIT déjà que len(labels) >= 1, sinon la tranche
        # n'aurait pas été écrite dans le CSV.

        # Masques binaires par label
        num_objs = len(labels)
        H, W = mask_tensor.shape
        masks = torch.zeros((num_objs, H, W), dtype=torch.uint8)
        for i, lbl in enumerate(labels):
            masks[i] = (mask_tensor == lbl).byte()

        # Bounding boxes
        boxes = masks_to_boxes(masks)  # shape [num_objs, 4]
        # Pas de check, on suppose la bbox est OK (sinon c'est un bug d'extraction).

        # area, iscrowd
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            scan_tensor = self.transforms(scan_tensor)

        target = {
            "boxes": boxes,
            "masks": masks,
            "labels": labels.long(),
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd
        }

        return scan_tensor, target


############################################################
# (C) COLLATE_FN (si on veut en avoir une spécifique)
############################################################
def collate_remove_none(batch):
    """
    Si vous n'avez plus JAMAIS de None, vous pouvez utiliser utils.collate_fn directement.
    Mais on peut garder ceci en 'backup' : 
    """
    batch = [b for b in batch if b is not None]
    return utils.collate_fn(batch)


############################################################
# (D) MODÈLE + MÉTRIQUES
############################################################
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    # Remplacer la tête box
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Remplacer la tête mask
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def generate_confusion_matrix(data_loader_test, model, device, num_classes, output_folder):
    all_true_labels = []
    all_pred_labels = []

    model.eval()
    for images, targets in data_loader_test:
        # Envoi sur GPU
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)
        # On traite juste le premier du batch
        pred = outputs[0]
        true = targets[0]

        if len(pred['labels']) > 0:
            # label prédit = le plus confiant
            idx_best = pred['scores'].argmax()
            pred_label = pred['labels'][idx_best].item()
        else:
            pred_label = 0  # ou un code "pas d'objet"

        # On suppose qu'il y a au moins 1 label dans le target
        true_label = true['labels'][0].item()

        all_true_labels.append(true_label)
        all_pred_labels.append(pred_label)

    # Matrice de conf
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    conf_mat = confusion_matrix(all_true_labels, all_pred_labels, labels=range(num_classes))
    conf_mat_pct = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)

    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_pct, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (%)")
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.close()
    print("Confusion matrix saved in", output_folder)


############################################################
# (E) FONCTION D'ENTRAÎNEMENT PRINCIPALE
############################################################
def train_model(
    scans_folder,
    masks_folder,
    temp_folder,
    slices_csv,
    num_epochs,
    lr,
    device,
    output_folder
):
    # 1) Extraction
    print("=== Étape 1 : Extraction & sauvegarde des slices ===")
    extract_and_save_slices(
        scans_folder,
        masks_folder,
        temp_folder,
        slices_csv,
        min_box_size=185  # Ajustez si besoin
    )

    # 2) Dataset sans None
    print("=== Étape 2 : Création du Dataset sans None ===")
    dataset_full = TempSlicesDataset(slices_csv, transforms=None)
    full_size = len(dataset_full)
    print(f"[INFO] dataset_full size: {full_size}")

    # 3) Split train / val / test
    indices = np.arange(full_size)
    np.random.shuffle(indices)
    train_split = int(0.8 * full_size)
    val_split   = int(0.9 * full_size)
    train_idx = indices[:train_split]
    val_idx   = indices[train_split:val_split]
    test_idx  = indices[val_split:]

    train_dataset = torch.utils.data.Subset(dataset_full, train_idx)
    val_dataset   = torch.utils.data.Subset(dataset_full, val_idx)
    test_dataset  = torch.utils.data.Subset(dataset_full, test_idx)

    print("=== Étape 3 : DataLoaders ===")
    # Puisque on ne renvoie plus None, on peut utiliser utils.collate_fn.
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=0
    )

    # 4) Modèle
    print("=== Étape 4 : Instanciation du modèle ===")
    num_classes = 4  # Fond + 3 classes par ex.
    model = get_model_instance_segmentation(num_classes).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 5) Entraînement
    print("=== Étape 5 : Entraînement ===")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        evaluate(model, val_loader, device=device)
        lr_scheduler.step()

    os.makedirs(output_folder, exist_ok=True)
    model_path = os.path.join(output_folder, "final_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Modèle final sauvegardé : {model_path}")

    # 6) Matrice de confusion
    print("=== Étape 6 : Évaluation Test & Matrice de confusion ===")
    model.eval()
    generate_confusion_matrix(
        test_loader,
        model,
        device,
        num_classes,
        output_folder
    )

    # 7) Évaluation finale
    evaluate(model, test_loader, device=device)

    return model_path


############################################################
# (F) MAIN
############################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN model with valid NIfTI slices (extracted to .npy)."
    )
    parser.add_argument('--scans_folder', type=str, required=True,
                        help='Répertoire contenant les fichiers scan .nii.')
    parser.add_argument('--masks_folder', type=str, required=True,
                        help='Répertoire contenant les fichiers masque .nii.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Nombre total d\'époques.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Dossier pour sauvegarder le modèle et résultats.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--temp_folder', type=str, default='temp_slices',
                        help='Dossier pour stocker les slices .npy.')
    parser.add_argument('--csv_path', type=str, default='slices_metadata.csv',
                        help='Fichier CSV listant les slices.')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    train_model(
        scans_folder=args.scans_folder,
        masks_folder=args.masks_folder,
        temp_folder=args.temp_folder,
        slices_csv=args.csv_path,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_folder=args.output_folder
    )
