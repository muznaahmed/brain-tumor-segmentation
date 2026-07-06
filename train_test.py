import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model import UNet

class BrainTumorDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx]).unsqueeze(0).float()  # add channel dim
        mask = torch.tensor(self.masks[idx]).unsqueeze(0).float()
        return img, mask

images = np.load("processed_data/images.npy")
masks = np.load("processed_data/masks.npy")

dataset = BrainTumorDataset(images, masks)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss()

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, masks_batch in train_loader:
        imgs, masks_batch = imgs.to(device), masks_batch.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, masks_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, avg training loss: {total_loss/len(train_loader):.4f}")

print("\nPipeline test complete — training loop runs successfully.")