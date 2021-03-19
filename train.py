import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Yolov1
from loss import YoloLoss
from dataset import VOCDataset

seed = 123
torch.manual_seed(seed)


LEARNING_RATE = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = 'overfit.pth.tar'
IMG_DIR = 'data/images'
LABEL_DIR = 'data/labels'


# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms
#

# transforms = Compose(
#     [transforms.Resize(448, 448),
#      transforms.ToTensor(),]
# )


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item)
    print(f'mean loss was {sum(mean_loss)/len(mean_loss)}')


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        pass

    train_transform = transforms.Compose([
        transforms.Resize(size=(448, 448)),
        transforms.ToTensor()
    ])
    train_dataset = VOCDataset(csv_file='', img_root=IMG_DIR, S=7, B=2, C=20, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(size=(448, 448)),
        transforms.ToTensor()
    ])
    test_dataset = VOCDataset(
        csv_file='',
        img_root=IMG_DIR,
        transform=test_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(EPOCHS):
        train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn
        )