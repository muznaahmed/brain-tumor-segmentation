import nibabel as nib
import numpy as np
import torch
import os
import random
from model import UNet

base_dir = "data_subset_45/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
patient_folders = sorted(os.listdir(base_dir))

# Reproduce the EXACT same train/test split as preprocess_split.py (same seed)
random.seed(42)
shuffled = patient_folders.copy()
random.shuffle(shuffled)
split_point = int(0.8 * len(shuffled))
test_patients = shuffled[split_point:]

print(f"Estimating tumor volume for {len(test_patients)} test patients:\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("unet_brain_tumor.pth", map_location=device))
model.eval()

voxel_volume_mm3 = 1.0  # BraTS is resampled to 1mm x 1mm x 1mm isotropic voxels

results = []

with torch.no_grad():
    for patient in test_patients:
        patient_path = os.path.join(base_dir, patient)
        flair_path = os.path.join(patient_path, f"{patient}_flair.nii")
        seg_path = os.path.join(patient_path, f"{patient}_seg.nii")

        flair = nib.load(flair_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        brain_voxels = flair[flair > 0]
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        flair_standardized = (flair - mean) / std

        ground_truth_mask = (seg > 0).astype(np.float32)
        ground_truth_voxels = ground_truth_mask.sum()

        predicted_voxel_count = 0
        num_slices = flair.shape[2]
        for i in range(num_slices):
            img_slice = torch.tensor(flair_standardized[:, :, i]).unsqueeze(0).unsqueeze(0).float().to(device)
            pred = model(img_slice)
            pred_binary = (pred > 0.5).float().cpu().numpy()[0, 0]
            predicted_voxel_count += pred_binary.sum()

        gt_volume_mm3 = ground_truth_voxels * voxel_volume_mm3
        pred_volume_mm3 = predicted_voxel_count * voxel_volume_mm3
        percent_error = abs(pred_volume_mm3 - gt_volume_mm3) / gt_volume_mm3 * 100

        results.append({
            "patient": patient,
            "gt_volume_mm3": gt_volume_mm3,
            "pred_volume_mm3": pred_volume_mm3,
            "percent_error": percent_error
        })

        print(f"{patient}: ground truth = {gt_volume_mm3:,.0f} mm³, "
              f"predicted = {pred_volume_mm3:,.0f} mm³, "
              f"error = {percent_error:.1f}%")

avg_error = np.mean([r["percent_error"] for r in results])
print(f"\nAverage volume estimation error across test patients: {avg_error:.1f}%")
median_error = np.median([r["percent_error"] for r in results])
print(f"Median volume estimation error across test patients: {median_error:.1f}%")