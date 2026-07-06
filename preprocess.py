import nibabel as nib
import numpy as np
import os

base_dir = "data_subset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

patient_folders = sorted(os.listdir(base_dir))

all_images = []
all_masks = []

for patient in patient_folders:
    patient_path = os.path.join(base_dir, patient)
    flair_path = os.path.join(patient_path, f"{patient}_flair.nii")
    seg_path = os.path.join(patient_path, f"{patient}_seg.nii")

    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    # Standardize using only brain tissue (nonzero voxels), not the black background
    brain_voxels = flair[flair > 0]
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    flair_standardized = (flair - mean) / std

    # Binary mask: 1 = any tumor label (1, 2, or 4), 0 = background/healthy tissue
    binary_mask = (seg > 0).astype(np.float32)

    # Keep only slices that contain at least some tumor
    num_slices = flair.shape[2]
    kept_this_patient = 0
    for i in range(num_slices):
        if binary_mask[:, :, i].sum() > 0:
            all_images.append(flair_standardized[:, :, i])
            all_masks.append(binary_mask[:, :, i])
            kept_this_patient += 1

    print(f"{patient}: kept {kept_this_patient} tumor-containing slices out of {num_slices}")

all_images = np.array(all_images, dtype=np.float32)
all_masks = np.array(all_masks, dtype=np.float32)

print(f"\nFinal dataset shape: images {all_images.shape}, masks {all_masks.shape}")

np.save(os.path.join(output_dir, "images.npy"), all_images)
np.save(os.path.join(output_dir, "masks.npy"), all_masks)
print(f"Saved to {output_dir}/images.npy and {output_dir}/masks.npy")