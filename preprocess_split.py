import nibabel as nib
import numpy as np
import os
import random

base_dir = "data_subset_45/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

patient_folders = sorted(os.listdir(base_dir))
print(f"Total patients found: {len(patient_folders)}")

# Split patients BEFORE extracting slices, so no patient leaks across train/test
random.seed(42)  # fixed seed = reproducible split every time you rerun this
shuffled = patient_folders.copy()
random.shuffle(shuffled)

split_point = int(0.8 * len(shuffled))
train_patients = shuffled[:split_point]
test_patients = shuffled[split_point:]

print(f"Train patients: {len(train_patients)}")
print(f"Test patients: {len(test_patients)}")

def process_patients(patient_list, label):
    images = []
    masks = []
    for patient in patient_list:
        patient_path = os.path.join(base_dir, patient)
        flair_path = os.path.join(patient_path, f"{patient}_flair.nii")
        seg_path = os.path.join(patient_path, f"{patient}_seg.nii")

        flair = nib.load(flair_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        brain_voxels = flair[flair > 0]
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        flair_standardized = (flair - mean) / std

        binary_mask = (seg > 0).astype(np.float32)

        for i in range(flair.shape[2]):
            if binary_mask[:, :, i].sum() > 0:
                images.append(flair_standardized[:, :, i])
                masks.append(binary_mask[:, :, i])

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    print(f"{label}: {images.shape[0]} tumor-containing slices from {len(patient_list)} patients")
    return images, masks

train_images, train_masks = process_patients(train_patients, "Train")
test_images, test_masks = process_patients(test_patients, "Test")

np.save(os.path.join(output_dir, "train_images.npy"), train_images)
np.save(os.path.join(output_dir, "train_masks.npy"), train_masks)
np.save(os.path.join(output_dir, "test_images.npy"), test_images)
np.save(os.path.join(output_dir, "test_masks.npy"), test_masks)

print(f"\nSaved train/test arrays to {output_dir}/")