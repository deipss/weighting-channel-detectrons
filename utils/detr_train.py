import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets.brackish import BrackishDataset
from model.detr import DETR
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 1024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class DETRLoss(nn.Module):
    def __init__(self):
        super(DETRLoss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.L1Loss()

    def forward(self, class_out, bbox_out, labels, bboxes):
        class_loss = self.class_loss(class_out, labels)
        bbox_loss = self.bbox_loss(bbox_out, bboxes)
        return class_loss + bbox_loss


def collate_fn_tuple(x):
    return tuple(zip(*x))


def train():
    root_dir = '/Users/deipss/workspace/ai/weighting-channel-detectrons/data/Brackish/dataset/img'
    train_annotation_file = '/Users/deipss/workspace/ai/weighting-channel-detectrons/data/Brackish/annotations/annotations_COCO/train_groundtruth.json'
    val_annotation_file = '/Users/deipss/workspace/ai/weighting-channel-detectrons/data/Brackish/annotations/annotations_COCO/valid_groundtruth.json'

    train_dataset = BrackishDataset(root_dir, train_annotation_file)
    val_dataset = BrackishDataset(root_dir, val_annotation_file)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2,
                              collate_fn=collate_fn_tuple)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2,
                            collate_fn=collate_fn_tuple)

    loss_fn = DETRLoss()

    model = DETR(num_classes=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    loss_fn = loss_fn.to(device)

    num_epochs = 10
    print_freq = 10
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels, bboxes) in enumerate(train_loader):
            images = torch.stack(images).to(device)
            labels = torch.tensor(labels).to(device)
            bboxes = torch.tensor(bboxes).to(device)

            optimizer.zero_grad()
            class_out, bbox_out = model(images, labels)
            loss = loss_fn(class_out, bbox_out, labels, bboxes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % print_freq == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(save_dir, f'model_{epoch}.pth'))

        # 验证模型
        model.eval()
        with torch.no_grad():
            for images, labels, bboxes in val_loader:
                images = torch.stack(images).to(device)
                labels = torch.tensor(labels).to(device)
                bboxes = torch.tensor(bboxes).to(device)

                class_out, bbox_out = model(images, labels)
                loss = loss_fn(class_out, bbox_out, labels, bboxes)
                print(f'Validation Loss: {loss.item():.4f}')


if __name__ == '__main__':
    train()
