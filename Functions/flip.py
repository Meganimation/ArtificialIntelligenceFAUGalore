import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# custom dataset for the flip pretext task
class FlipPretextDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Grab the original image (ignore the original label)
        img, _ = self.base_dataset[idx]
        
        # Randomly choose a transformation (0-3)
        target = torch.randint(0, 4, (1,)).item()
        
        if target == 1:   # Horizontal Flip
            img = TF.hflip(img)
        elif target == 2: # Vertical Flip
            img = TF.vflip(img)
        elif target == 3: # Both
            img = TF.hflip(TF.vflip(img))
            
        return img, target

# setup
transform = transforms.Compose([transforms.ToTensor()])
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
pretext_loader = DataLoader(FlipPretextDataset(cifar10), batch_size=32, shuffle=True)

# this is the model im using
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4) # 4 classes: None, H, V, Both

# this is the part where we need to adjust the first conv layer and remove the maxpool to preserve spatial info for the flip task
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# the training loop
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def train_one_epoch():
    model.train()
    total = len(pretext_loader)
    for i, (images, targets) in enumerate(pretext_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"Batch [{i+1}/{total}] Loss: {loss.item():.4f}")

train_one_epoch()