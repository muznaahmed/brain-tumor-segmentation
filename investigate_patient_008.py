import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import UNet

patient = "BraTS20_Training_008"
patient_path = f"data_subset_45/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/{patient}"

flair = nib.load(f"{patient_path}/{patient}_flair.nii").get_fdata()
seg = nib.load(f"{patient_path}/{patient}_seg.nii").get_fdata()

brain_voxels = flair[flair > 0]
mean = brain_voxels.mean()
std = brain_voxels.std()
flair_standardized = (flair - mean) / std

ground_truth_mask = (seg > 0).astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("unet_brain_tumor.pth", map_location=device))
model.eval()

num_slices = flair.shape[2]
false_positive_per_slice = []

with torch.no_grad():
    for i in range(num_slices):
        img_slice = torch.tensor(flair_standardized[:, :, i]).unsqueeze(0).unsqueeze(0).float().to(device)
        pred = model(img_slice)
        pred_binary = (pred > 0.5).float().cpu().numpy()[0, 0]

        gt_slice = ground_truth_mask[:, :, i]
        # False positives: predicted tumor where there is NO ground truth tumor
        false_positives = ((pred_binary == 1) & (gt_slice == 0)).sum()
        false_positive_per_slice.append(false_positives)

false_positive_per_slice = np.array(false_positive_per_slice)

# Find the 3 worst slices (most false-positive pixels)
worst_slices = np.argsort(false_positive_per_slice)[-3:][::-1]
print("Worst slices (most false-positive pixels):")
for s in worst_slices:
    print(f"  Slice {s}: {int(false_positive_per_slice[s])} false-positive pixels")

fig, axes = plt.subplots(len(worst_slices), 3, figsize=(12, 4 * len(worst_slices)))

with torch.no_grad():
    for row, s in enumerate(worst_slices):
        img_slice = torch.tensor(flair_standardized[:, :, s]).unsqueeze(0).unsqueeze(0).float().to(device)
        pred = model(img_slice)
        pred_binary = (pred > 0.5).float().cpu().numpy()[0, 0]

        axes[row, 0].imshow(flair[:, :, s], cmap="gray")
        axes[row, 0].set_title(f"FLAIR (slice {s})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(ground_truth_mask[:, :, s], cmap="gray")
        axes[row, 1].set_title("Ground truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_binary, cmap="gray")
        axes[row, 2].set_title(f"Predicted ({int(false_positive_per_slice[s])} FP pixels)")
        axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("patient_008_investigation.png", dpi=150)
print("\nSaved to patient_008_investigation.png")