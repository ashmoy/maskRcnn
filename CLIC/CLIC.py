import os
import slicer
import qt
import webbrowser
import ctk
from slicer.ScriptedLoadableModule import *
import torch
import subprocess
import numpy as np
import nibabel as nib
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import scipy.ndimage
import time
from CondaSetUp import CondaSetUpCall, CondaSetUpCallWsl
import threading
from slicer.util import VTKObservationMixin
import logging
import requests
import glob
import zipfile

def PathFromNode(node):
  storageNode = node.GetStorageNode()
  if storageNode is not None:
    filepath = storageNode.GetFullNameFromFileName()
  else:
    filepath = None
  return filepath

# -- Classe pour les signaux de mise à jour --
class ProgressUpdater(qt.QObject):
  progressChanged = qt.Signal(int)
  logChanged = qt.Signal(str)

class CLIC(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "CLIC"
    self.parent.categories = ["Segmentation"]
    self.parent.dependencies = ["CondaSetUp"]
    self.parent.contributors = ["Your Name"]
    self.parent.helpText = """This module performs segmentation using a Mask R-CNN model."""
    self.parent.acknowledgementText = "Thanks to the community for support."

class CLICWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

    self.CBCT_as_input = True   # True: CBCT image, False: surface IOS
    self.folder_as_input = False
    self.MRMLNode_scan = None
    self.input_path = None
    self.model_folder = None
    self.output_folder = None
    self.goup_output_files = False
    self.scan_count = 0
    self.currentSegNode = None  # Pour suivre le volume affiché

  def UpdateSaveType(self, checked):
    self.goup_output_files = checked
    self.ui.SearchSaveFolder.setHidden(checked)
    self.ui.SaveFolderLineEdit.setHidden(checked)
    self.ui.PredictFolderLabel.setHidden(checked)

  def setup(self):
    self.conda_wsl = CondaSetUpCallWsl()
    ScriptedLoadableModuleWidget.setup(self)

    uiWidget = slicer.util.loadUI(self.resourcePath('UI/CLIC.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Création de la logique métier
    self.logic = CLICLogic()

    # Instanciation de l'updater pour les signaux
    self.progressUpdater = ProgressUpdater()
    # La progressBar "PredScanProgressBar" sera utilisée pour le téléchargement,
    # et "progressBar" pour le traitement des slices.
    self.progressUpdater.progressChanged.connect(lambda value: self.ui.progressBar.setValue(value))
    self.progressUpdater.logChanged.connect(lambda text: self.ui.logTextEdit.append(text))

    self.ui.SavePredictCheckBox.toggled.connect(self.UpdateSaveType)
    self.ui.DownloadModelPushButton.clicked.connect(self.onModelDownloadButton)
    self.ui.SearchScanFolder.clicked.connect(self.onSearchScanButton)
    self.ui.SearchModelFolder.clicked.connect(self.onSearchModelButton)
    self.ui.SearchSaveFolder.clicked.connect(self.onSearchSaveButton)
    self.ui.PredictionButton.clicked.connect(self.onPredictButton)
    self.ui.CancelButton.clicked.connect(self.onCancel)

    self.ui.progressBar.setVisible(False)
    self.RunningUI(False)
    self.initializeParameterNode()

  def SwitchInputExtension(self, index):
    if index == 1:
      self.ui.ScanPathLabel.setText('DICOM Folder')
    else:
      self.ui.ScanPathLabel.setText('Scan File/Folder')

  def onNodeChanged(self):
    self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
    if self.MRMLNode_scan:
      self.input_path = PathFromNode(self.MRMLNode_scan)
      self.scan_count = 1
      self.ui.PrePredInfo.setText("Number of scans to process: 1")
      return True
    return False

  def onModelDownloadButton(self):
    model_url = "https://github.com/ashmoy/maskRcnn/releases/download/model/final_model.pth"
    default_model_folder = os.path.join(os.path.expanduser("~"), "Documents", "CLIC_Models")
    os.makedirs(default_model_folder, exist_ok=True)
    model_path = os.path.join(default_model_folder, "final_model.pth")
    try:
      self.ui.PredScanLabel.setText("Downloading model...")
      response = requests.get(model_url, stream=True)
      response.raise_for_status()
      total_size = int(response.headers.get('content-length', 0))
      downloaded_size = 0
      chunk_size = 1024
      with open(model_path, 'wb') as file:
        for data in response.iter_content(chunk_size=chunk_size):
          file.write(data)
          downloaded_size += len(data)
          progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
          self.ui.PredScanProgressBar.setValue(int(progress))
      self.ui.lineEditModelPath.setText(default_model_folder)
      self.model_folder = default_model_folder
      self.ui.PredScanLabel.setText("Model downloaded successfully!")
    except Exception as e:
      qt.QMessageBox.warning(self.parent, 'Error', f"Failed to download model: {str(e)}")
      self.ui.PredScanLabel.setText("Error during download.")

  def onSearchScanButton(self):
    scan_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
    if scan_folder:
      self.input_path = scan_folder
      self.ui.lineEditScanPath.setText(self.input_path)

  def onSearchModelButton(self):
    model_path = qt.QFileDialog.getExistingDirectory(self.parent, "Select Model Folder", "")
    if model_path:
      self.model_folder = model_path
      self.ui.lineEditModelPath.setText(model_path)

  def onSearchSaveButton(self):
    save_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select Save Folder", "")
    if save_folder:
      self.output_folder = save_folder
      self.ui.SaveFolderLineEdit.setText(save_folder)

  def onPredictButton(self):
    if not self.input_path:
      qt.QMessageBox.warning(self.parent, 'Warning', 'Please select an input file/folder')
      return
    if not self.model_folder:
      qt.QMessageBox.warning(self.parent, 'Warning', 'Please select a model folder')
      return

    param = {
      "input_path": self.input_path,
      "model_folder": self.model_folder,
      "output_dir": self.output_folder,
    }

    # Définition des callbacks pour progression, log et affichage.
    # Remplacement de invokeLater par qt.QTimer.singleShot pour l'affichage dans le thread principal.
    def update_progress(progress):
      self.progressUpdater.progressChanged.emit(int(progress))
    def update_log(message):
      self.progressUpdater.logChanged.emit(message)
    def display_segmentation(seg_file):
      qt.QTimer.singleShot(0, lambda: self.load_segmentation(seg_file))

    try:
      nii_files = glob.glob(os.path.join(self.input_path, "*.nii"))
      if nii_files:
        self.load_nii_in_slicer(nii_files[0])
      self.RunningUI(True)
      self.processThread = threading.Thread(target=self.logic.process, args=(param, update_progress, update_log, display_segmentation))
      self.processThread.start()
      start_time = time.time()
      while self.processThread.is_alive():
        slicer.app.processEvents()
        current_time = time.time()
        self.ui.TimerLabel.setText(f"Time elapsed: {current_time - start_time:.2f}s")
        time.sleep(0.1)
      qt.QMessageBox.information(self.parent, 'Success', 'Segmentation completed successfully!')
    except Exception as e:
      qt.QMessageBox.warning(self.parent, 'Error', f'An error occurred during segmentation: {str(e)}')
    finally:
      self.RunningUI(False)

  def load_segmentation(self, seg_file):
    if self.currentSegNode:
      slicer.mrmlScene.RemoveNode(self.currentSegNode)
      self.currentSegNode = None
    loadedNodes = slicer.util.loadVolume(seg_file)
    if isinstance(loadedNodes, dict) and "volumeNode" in loadedNodes:
      self.currentSegNode = loadedNodes["volumeNode"]

  def onCancel(self):
    if self.logic:
      self.logic.cancelRequested = True
      self.progressUpdater.logChanged.emit("Cancellation requested.")
    if hasattr(self, "processThread") and self.processThread.is_alive():
      self.processThread.join(1)
      self.ui.PredScanLabel.setText("Prediction canceled.")

  def RunningUI(self, running):
    self.ui.PredictionButton.setEnabled(not running)
    self.ui.CancelButton.setEnabled(running)
    self.ui.progressBar.setVisible(running)

  def initializeParameterNode(self):
    pass

  def load_nii_in_slicer(self, nii_file):
    if not os.path.exists(nii_file):
      raise FileNotFoundError(f"NIfTI file not found: {nii_file}")
    volume_node = slicer.util.loadVolume(nii_file)
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.setSliceViewerLayers(background=volume_node)

class CLICLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.seg_files = []  # Pour stocker les segmentations sauvegardées
    self.cancelRequested = False

  CLASS_NAMES = {1: "buccal", 2: "bicortical", 3: "palatal"}

  def get_model_instance_segmentation(self, num_classes):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

  def load_model(self, model_path, num_classes, device):
    model = self.get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

  def process_nii_file(self, model, nii_path, device, progress_callback=None, log_callback=None, score_threshold=0.5):
    nib_vol = nib.load(nii_path)
    vol_data = nib_vol.get_fdata(dtype=np.float32)
    H, W, Z = vol_data.shape
    if log_callback:
      log_callback("Processing slices for file: " + nii_path)
    all_detections = []
    slice_counts = {"buccal": {"left": 0, "right": 0},
                    "bicortical": {"left": 0, "right": 0},
                    "palatal": {"left": 0, "right": 0}}
    for z in range(Z):
      if self.cancelRequested:
        if log_callback:
          log_callback("Processing cancelled during slice processing at slice %d." % z)
        break
      slice_2d = vol_data[..., z]
      slice_norm = self.normalize_slice(slice_2d)
      slice_tensor = torch.from_numpy(slice_norm).unsqueeze(0).repeat(3, 1, 1).float().to(device)
      with torch.no_grad():
        preds = model([slice_tensor])[0]
      scores = preds["scores"]
      masks = preds.get("masks", None)
      labels = preds["labels"]
      keep = scores >= score_threshold
      scores = scores[keep]
      labels = labels[keep]
      if masks is not None:
        masks = masks[keep]
      if masks is None or len(scores) == 0:
        if progress_callback:
          progress = ((z + 1) / Z) * 100
          progress_callback(progress)
        continue
      labels_np = labels.cpu().numpy()
      masks_np = (masks > 0.5).squeeze(1).cpu().numpy()
      for i in range(len(masks_np)):
        l = int(labels_np[i])
        mk = masks_np[i]
        com = scipy.ndimage.center_of_mass(mk)
        side = "left" if com[0] < (H/2) else "right"
        slice_counts[self.CLASS_NAMES[l]][side] += 1
        all_detections.append({
          "label": l,
          "slice_z": z,
          "mask_2d": mk
        })
      if progress_callback:
        progress = ((z + 1) / Z) * 100
        progress_callback(progress)
    return vol_data, nib_vol, all_detections, slice_counts

  def normalize_slice(self, slice_2d):
    mn, mx = slice_2d.min(), slice_2d.max()
    if (mx - mn) > 1e-8:
      return (slice_2d - mn) / (mx - mn)
    else:
      return np.zeros_like(slice_2d, dtype=np.float32)

  def save_nii(self, vol_np, nib_ref, out_path):
    nii_img = nib.Nifti1Image(vol_np.astype(np.int16), nib_ref.affine, nib_ref.header)
    nib.save(nii_img, out_path)

  def process(self, parameters, progress_callback=None, log_callback=None, display_callback=None):
    try:
      self.cancelRequested = False
      input_path = parameters["input_path"]
      model_folder = parameters["model_folder"]
      output_dir = parameters["output_dir"]

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model_files = glob.glob(os.path.join(model_folder, "*.pth"))
      if not model_files:
        raise FileNotFoundError("No .pth files found in the specified model folder.")
      model_path = model_files[0]
      num_classes = 4
      model = self.load_model(model_path, num_classes, device)

      if os.path.isfile(input_path) and (input_path.endswith(".nii") or input_path.endswith(".nii.gz")):
        file_list = [input_path]
      elif os.path.isdir(input_path):
        file_list = glob.glob(os.path.join(input_path, "*.nii")) + glob.glob(os.path.join(input_path, "*.nii.gz"))
      else:
        raise ValueError("Invalid input path. Please provide a valid .nii/.nii.gz file or folder.")

      seg_files = []
      for nii_file in file_list:
        if self.cancelRequested:
          if log_callback:
            log_callback("Processing cancelled before file: " + nii_file)
          break
        if log_callback:
          log_callback("Processing file: " + nii_file)
        vol_data, nib_ref, detections, counts = self.process_nii_file(model, nii_file, device, progress_callback, log_callback)
        if log_callback:
          log_callback("Finished processing file: " + nii_file)
          log_callback("Summary for file " + os.path.basename(nii_file) + ":")
          for cls in counts:
            log_callback(f"  {cls}: Left = {counts[cls]['left']}, Right = {counts[cls]['right']}")
        seg_data = np.zeros(vol_data.shape, dtype=np.int16)
        for det in detections:
          z = det["slice_z"]
          mask = det["mask_2d"]
          label = det["label"]
          seg_data[..., z][mask] = label
        base_name = os.path.basename(nii_file).replace(".nii.gz", "").replace(".nii", "")
        output_filename = f"{base_name}_seg.nii.gz"
        output_path = os.path.join(output_dir, output_filename)
        self.save_nii(seg_data, nib_ref, output_path)
        seg_files.append(output_path)
        if display_callback:
          display_callback(output_path)
      if log_callback:
        log_callback("Processing completed successfully.")
      self.seg_files = seg_files
    except Exception as e:
      if log_callback:
        log_callback(f"Error during processing: {str(e)}")
      raise
