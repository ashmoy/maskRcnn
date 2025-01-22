import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.ops.boxes import masks_to_boxes
from torchvision import transforms as T
from torch import nn, optim
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import tv_tensors
import utils
from torchvision.transforms.v2 import functional as F
from engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from captum.attr import IntegratedGradients, GradientShap
# Vérifier la disponibilité du GPU
from monai.visualize import GradCAM
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'




def apply_gradcam_and_save_nii(model, test_loader, device, output_folder, baselines=None, steps=10):
    integrated_gradients = IntegratedGradients(model)
    print("Applying Integrated Gradients for attribution...")

    model.eval()  # Mettre le modèle en mode évaluation

    for batch_idx, (images, targets) in enumerate(test_loader):
        for img_idx, image in enumerate(images):
            # Vérifier la dimension de l'image
            if image.dim() == 3:  # Si c'est un volume 2D [C, H, W]
                print(f"Processing image {batch_idx}_{img_idx} with shape {image.shape}...")
                torch.cuda.empty_cache()  # Libérer la mémoire GPU

                # S'assurer que l'image est sur le bon appareil
                image = image.to(device)


                # Appliquer les gradients intégrés

                # Extraire le tensor de sortie pertinent si le modèle renvoie un dictionnaire
                output = model(image.unsqueeze(0))
                print(f"Output: {output}")

                if isinstance(output, list) and len(output) > 0:
                    output = output[0]

                # Extraire les scores, labels, et boîtes depuis la sortie du modèle
                scores = output['scores']
                labels = output['labels']
                print(f"Scores: {scores}")

                # Trouver l'indice du score le plus élevé
                _, max_score_idx = scores.max(0)

                # Récupérer le label correspondant
                target_label = labels[max_score_idx].item()
                print(f"Target label: {target_label}")
                image_tensor = torch.tensor(image).to(device) if not isinstance(image, torch.Tensor) else image.to(device)
                print(f"Image tensor shape: {image_tensor.shape}",'for ', image_tensor)

                # Vérifier si l'attribution peut être calculée
                if image.dim() == 3:  # Doit être au format (C, H, W)
                    # Appliquer Integrated Gradients
                    attributions = integrated_gradients.attribute(
                        inputs=image_tensor.unsqueeze(0),  # Ajouter la dimension batch
                        target=target_label,
                        n_steps=steps
                    )


                else:
                    print(f"L'image {batch_idx}_{img_idx} n'est pas au bon format pour l'attribution.")























#######################################
# Fonction pour extraire les slices valides
#######################################
def extract_valid_slices(scan_file, mask_file):
    try:
        scan_img = nib.load(scan_file)
        mask_img = nib.load(mask_file)
    except Exception as e:
        return [], []

    scan_data = scan_img.get_fdata()
    mask_data = mask_img.get_fdata()
    min_slices = min(scan_data.shape[2], mask_data.shape[2])

    valid_scan_slices = []
    valid_mask_slices = []

    for i in range(min_slices):
        mask_slice = mask_data[:, :, i]
        scan_slice = scan_data[:, :, i]
        y, x = torch.where(torch.tensor(mask_slice) != 0)

        if len(y) > 0:
            y_min, y_max = torch.min(y), torch.max(y)
            x_min, x_max = torch.min(x), torch.max(x)
            height = y_max - y_min
            width = x_max - x_min

            if height >= 35 and width >= 35:
                valid_scan_slices.append(scan_slice)
                valid_mask_slices.append(mask_slice)

    return valid_scan_slices, valid_mask_slices

#######################################
# Dataset personnalisé pour les slices valides
#######################################
class CanineNiftiValidSliceDataset(Dataset):
    def __init__(self, scan_folder, mask_folder, transforms=None):
        self.scan_folder = scan_folder
        self.mask_folder = mask_folder
        self.transforms = transforms
        self.scan_files = sorted([f for f in os.listdir(scan_folder) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.nii')])
        assert len(self.scan_files) == len(self.mask_files), "Mismatch between scans and masks!"

        self.valid_scan_slices = []
        self.valid_mask_slices = []

        for scan_file, mask_file in tqdm(zip(self.scan_files, self.mask_files), total=len(self.scan_files), desc="Extracting valid slices"):
            scan_path = os.path.join(self.scan_folder, scan_file)
            mask_path = os.path.join(self.mask_folder, mask_file)

            valid_scan_slices, valid_mask_slices = extract_valid_slices(scan_path, mask_path)
            self.valid_scan_slices.extend(valid_scan_slices)
            self.valid_mask_slices.extend(valid_mask_slices)

    def __len__(self):
        return len(self.valid_scan_slices)

    def __getitem__(self, idx):
        scan_slice = self.valid_scan_slices[idx]
        mask_slice = self.valid_mask_slices[idx]

        scan_slice = (scan_slice - np.min(scan_slice)) / (np.max(scan_slice) - np.min(scan_slice))
        scan_tensor = torch.from_numpy(scan_slice).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_slice).long()

        labels = torch.unique(mask_tensor)
        if len(labels) == 1 and labels[0] == 0:
            return None

        labels = labels[1:]
        num_objs = len(labels)

        height, width = mask_tensor.shape[-2:]
        masks = torch.zeros((num_objs, height, width), dtype=torch.uint8)
        for i, obj_id in enumerate(labels):
            masks[i] = (mask_tensor == obj_id).to(dtype=torch.uint8)

        boxes = masks_to_boxes(masks)

        valid_boxes, valid_labels, valid_masks = [], [], []
        for i, box in enumerate(boxes):
            if len(box) >= 0:
                if (box[2] > box[0]) and (box[3] > box[1]):
                    valid_boxes.append(box)
                    valid_labels.append(labels[i])
                    valid_masks.append(masks[i])

        if len(valid_boxes) == 0:
            return None

        boxes = torch.stack(valid_boxes)
        labels = torch.stack(valid_labels)
        masks = torch.stack(valid_masks)
        labels = labels.to(dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            scan_tensor = self.transforms(scan_tensor)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(scan_tensor)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd
        }

        return scan_tensor, target

#######################################
# Fonction pour obtenir le modèle Mask R-CNN personnalisé
#######################################
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

#######################################
# Fonction principale pour l'entraînement
#######################################
def train_model(scans_folder, masks_folder, num_epochs, lr, device, output_folder):
    # Définir les transformations
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5)
    ])

    # Charger les datasets avec les slices valides
    dataset = CanineNiftiValidSliceDataset(scans_folder, masks_folder, transforms=train_transforms)
    print('dataset ; termine')
    dataset_val = CanineNiftiValidSliceDataset(scans_folder, masks_folder, transforms=None)
    print('dataset_val ; termine')
    dataset_test = CanineNiftiValidSliceDataset(scans_folder, masks_folder, transforms=None)
    print('dataset_test ; termine')
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)

    train_split = int(0.8 * total_size)
    val_split = int(0.9 * total_size)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset_val, val_indices)
    test_dataset = torch.utils.data.Subset(dataset_test, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn, num_workers=32)

    # Initialiser le modèle
    num_classes = 4
    model = get_model_instance_segmentation(num_classes)
    model = model.to(device)

    # Définir l'optimiseur et le scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_val_loss = float('inf')  # Variable pour stocker la meilleure perte
    best_model_path = os.path.join(output_folder, 'best_model.pth')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)

        coco_evaluator = evaluate(model, val_loader, device=device)




    # Sauvegarde du dernier modèle
    model_save_path = os.path.join(output_folder, 'final_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

    # Générer la matrice de confusion pour le test
    model.eval()
    generate_confusion_matrix(test_loader, model, device, 4, get_transform(train=False), output_folder)

    apply_gradcam_and_save_nii(model, test_loader, device, output_folder)

    print(f"GradCAM attributions saved to {output_folder}")

    # Évaluation finale sur l'ensemble de test
    evaluate(model, test_loader, device=device)
    return best_model_path


#######################################
# Fonction pour définir les transformations
#######################################
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Lambda(lambda img: img.float()))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

# Générer la matrice de confusion
def generate_confusion_matrix(data_loader_test, model, device, num_classes, eval_transform, output_folder):
    all_true_labels = []
    all_pred_labels = []

    for images, targets in data_loader_test:
        images = list(img.to(device) for img in images)

        with torch.no_grad():
            if isinstance(images[0], torch.Tensor):
                x = images[0]
            else:
                x = eval_transform(images[0])
            x = x[:3, ...].to(device)
            predictions = model(x.unsqueeze(0))
            pred = predictions[0]

        true_labels = [target['labels'].cpu().numpy() for target in targets]

        scores = pred["scores"].cpu().numpy()
        sorted_indices = scores.argsort()[::-1].copy()

        for i, true in enumerate(true_labels):
            if len(true) == 1:
                top_label = pred["labels"][sorted_indices[0]].item()
                all_true_labels.append(true[0])
                all_pred_labels.append(top_label)
            elif len(true) == 2:
                top_labels = pred["labels"][sorted_indices[:2]].cpu().numpy().copy()
                all_true_labels.extend(true)
                all_pred_labels.extend(top_labels)

    valid_true_labels = [label for label in all_true_labels if label <= num_classes]
    valid_pred_labels = [label for label in all_pred_labels if label <= num_classes]

    if len(valid_true_labels) != len(valid_pred_labels):
        print(f"Error: Mismatch in the number of true and predicted labels.")
    else:
        conf_matrix = confusion_matrix(valid_true_labels, valid_pred_labels)
        conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Confusion Matrix:")
        print(conf_matrix_percentage)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_percentage, annot=True, cmap="Blues", fmt=".2f",
                    xticklabels=np.arange(num_classes),
                    yticklabels=np.arange(num_classes))
        plt.title("Confusion Matrix (Percentage)")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        output_path = os.path.join(output_folder, "confusion_matrix.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Confusion matrix saved to {output_path}")


#######################################
# Point d'entrée du script
#######################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Mask R-CNN model with valid NIfTI slices.")
    parser.add_argument('--scans_folder', type=str, required=True, help='Folder containing the scan .nii.gz files.')
    parser.add_argument('--masks_folder', type=str, required=True, help='Folder containing the mask .nii.gz files.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the model and results.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the optimizer.')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    train_model(args.scans_folder, args.masks_folder, args.epochs, args.lr, device, args.output_folder)

