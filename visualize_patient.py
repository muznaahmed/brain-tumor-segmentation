import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

patient_dir = "data_subset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001"

# Load the FLAIR scan (good for seeing tumor + edema) and the segmentation mask
flair_path = os.path.join(patient_dir, "BraTS20_Training_001_flair.nii")
seg_path = os.path.join(patient_dir, "BraTS20_Training_001_seg.nii")

flair_img = nib.load(flair_path).get_fdata()
seg_img = nib.load(seg_path).get_fdata()

print("FLAIR volume shape:", flair_img.shape)
print("Segmentation volume shape:", seg_img.shape)
print("Unique labels in segmentation:", np.unique(seg_img))

# Pick a middle slice (tumors are usually visible somewhere in the middle)
slice_index = flair_img.shape[2] // 2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(flair_img[:, :, slice_index], cmap="gray")
axes[0].set_title(f"FLAIR scan (slice {slice_index})")
axes[0].axis("off")

axes[1].imshow(flair_img[:, :, slice_index], cmap="gray")
axes[1].imshow(seg_img[:, :, slice_index], cmap="jet", alpha=0.4)
axes[1].set_title("FLAIR + tumor mask overlay")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("patient_001_check.png", dpi=150)
print("\nSaved visualization to patient_001_check.png")
plt.show()