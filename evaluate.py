import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import UNet
import matplotlib.pyplot as plt

class BrainTumorDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx]).unsqueeze(0).float()
        mask = torch.tensor(self.masks[idx]).unsqueeze(0).float()
        return img, mask

def dice_score(pred, target, epsilon=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

def iou_score(pred, target, epsilon=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)

test_images = np.load("processed_data/test_images.npy")
test_masks = np.load("processed_data/test_masks.npy")

test_ds = BrainTumorDataset(test_images, test_masks)
test_loader = DataLoader(test_ds, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("unet_brain_tumor.pth", map_location=device))
model.eval()

dice_scores = []
iou_scores = []

with torch.no_grad():
    for imgs, masks_batch in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs)
        preds_binary = (preds > 0.5).float().cpu().numpy()
        masks_np = masks_batch.numpy()

        for i in range(preds_binary.shape[0]):
            dice_scores.append(dice_score(preds_binary[i], masks_np[i]))
            iou_scores.append(iou_score(preds_binary[i], masks_np[i]))

print(f"Mean Dice score: {np.mean(dice_scores):.4f}")
print(f"Mean IoU score: {np.mean(iou_scores):.4f}")

# Visualize a few predictions vs ground truth
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
sample_indices = [0, len(test_images)//2, len(test_images)-1]

with torch.no_grad():
    for row, idx in enumerate(sample_indices):
        img = torch.tensor(test_images[idx]).unsqueeze(0).unsqueeze(0).float().to(device)
        pred = model(img)
        pred_binary = (pred > 0.5).float().cpu().numpy()[0, 0]

        axes[row, 0].imshow(test_images[idx], cmap="gray")
        axes[row, 0].set_title("FLAIR scan")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(test_masks[idx], cmap="gray")
        axes[row, 1].set_title("Ground truth mask")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_binary, cmap="gray")
        axes[row, 2].set_title("Predicted mask")
        axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("evaluation_samples.png", dpi=150)
print("Saved comparison to evaluation_samples.png")