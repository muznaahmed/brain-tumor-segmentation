import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import UNet
import time

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

train_images = np.load("processed_data/train_images.npy")
train_masks = np.load("processed_data/train_masks.npy")
test_images = np.load("processed_data/test_images.npy")
test_masks = np.load("processed_data/test_masks.npy")

train_ds = BrainTumorDataset(train_images, train_masks)
test_ds = BrainTumorDataset(test_images, test_masks)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss()

num_epochs = 15

for epoch in range(num_epochs):
    start_time = time.time()

    # Training
    model.train()
    train_loss = 0
    for imgs, masks_batch in train_loader:
        imgs, masks_batch = imgs.to(device), masks_batch.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, masks_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation (no gradient updates, just checking performance)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for imgs, masks_batch in test_loader:
            imgs, masks_batch = imgs.to(device), masks_batch.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks_batch)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} — train loss: {train_loss:.4f}, test loss: {test_loss:.4f} ({elapsed:.1f}s)")

torch.save(model.state_dict(), "unet_brain_tumor.pth")
print("\nModel saved to unet_brain_tumor.pth")