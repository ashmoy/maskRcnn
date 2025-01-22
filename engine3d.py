import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval3d import CocoEvaluator
from coco_utils3d import get_coco_api_from_dataset


from tqdm import tqdm

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    data_loader = tqdm(data_loader, desc=header)

    for i, slice_set in enumerate(data_loader):
        if slice_set is None:
            continue
        # Check if slice_set contains valid slices
        if not isinstance(slice_set, tuple) or len(slice_set) != 2:
            continue  # Skip invalid slice sets

        images, targets = slice_set

        # Ensure there are valid images and targets
        if images is None or targets is None or len(images) == 0 or len(targets) == 0:
            continue

        images_converted = []
        for img in images:
            # If img is a list of slices, process each slice
            if isinstance(img, list):
                slice_converted = []
                for slice_img in img:
                    if not isinstance(slice_img, torch.Tensor):
                        slice_img = torch.tensor(slice_img)  # Convert to tensor if needed
                    slice_img = slice_img.to(device)  # Move to the device
                    slice_converted.append(slice_img)
                images_converted.append(slice_converted)  # Add the processed slices as a list
            else:
                # If img is already a tensor, move to the device directly
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                img = img.to(device)
                images_converted.append(img)

        # Move targets to the appropriate device

        for target in targets:

            target = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in target]
            #print(f'targets: {targets}')

        for img, target in zip(images_converted, targets):
            if img is None or target is None:
                continue  # Skip if any invalid img or target is found


            target = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in target]


            loss_dict = model(img, target)

            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
                    # Reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # Check for NaN or Inf losses
            if torch.isnan(losses) or torch.isinf(losses):
                #print(f"Skipping batch due to invalid loss: {losses}")
                continue

            # Backward pass and optimization step
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # Update tqdm description to show the current loss
            data_loader.set_postfix(loss=f"{losses_reduced.item():.4f}")
            if i % print_freq == 0:
                metric_logger.log_every(i, print_freq, header)

    # Check if there was any valid data processed to avoid division by zero
    if metric_logger.meters['loss'].count > 0:
        print(f"Finished Epoch {epoch}, avg loss: {metric_logger.meters['loss'].global_avg}")
    else:
        print(f"Finished Epoch {epoch}, no valid data processed.")

    return metric_logger









def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types



@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image_set in metric_logger.log_every(data_loader, 100, header):
        if image_set is None or len(image_set) != 2:
            continue  # Ensure valid image-target pairs

        images, targets = image_set
        imags =[]
        for imgs in images:
            imags.append(list(img.to(device) for img in imgs))

            # if images is None or not isinstance(images, list) or len(images) == 0:
            #     print("Skipping invalid images in evaluation")
            #     continue
            # if targets is None or not isinstance(targets, list) or len(targets) == 0:
            #     print("Skipping invalid targets in evaluation")
            #     continue

            # # Validate the image format and dimensions
            # #valid_images = [img.to(device) for img in images if isinstance(img, torch.Tensor) and img.ndim == 3]

            # if len(images) == 0:
            #     print("No valid images for evaluation, skipping batch.")
            #     continue

        # Convert outputs to CPU for evaluation
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        out= []
        for images in imags:
            out.append(model(images))

        # Move outputs to CPU
        for output in out:
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        model_time = time.time() - model_time
        for t in targets:
            res = {target["image_id"]: output for target, output in zip(t, outputs)}
            print(f'res: {res}')
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


