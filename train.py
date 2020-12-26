from torch.utils.data import DataLoader
from crnn.data.dataset import (
    TextDataset,
    ToTensor,
    ZeroMean,
    Rescale,
    Gray,
    RandomConvert,
)
from torch.nn import CTCLoss, init
from torch import optim
from crnn.models.crnn import CRNN
from crnn.utils import ctc_decode
import torch
from crnn.config import opt
from torchvision import transforms
import time
import warnings
import os
from tqdm import tqdm, trange
import gc

warnings.filterwarnings("ignore")


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_out")
        init.constant_(m.bias, 0)


train_dataset = TextDataset(
    opt.train_filename,
    opt.root_dir,
    opt.max_label_length,
    transforms.Compose([Rescale((32, 100)), Gray(), ZeroMean(), ToTensor()]),
)

train_loader = DataLoader(train_dataset, 64, True)


device = opt.device
net = CRNN()
net.apply(weights_init)
net = net.to(device)
net.zero_grad()

params = net.parameters()

ctc_loss = CTCLoss()
optimizer = optim.Adam(params, weight_decay=1e-5)
best_loss = 50

print("gc is enabled", gc.isenabled())

for epoch in trange(opt.epoch):
    running_loss = 0.0
    for i, train_data in tqdm(enumerate(train_loader, 0)):
        inputs, labels, labels_length = (
            train_data["image"],
            train_data["label"],
            train_data["label_length"],
        )

        preds = net(inputs.to(device))
        optimizer.zero_grad()
        pred_labels = ctc_decode(preds)

        log_preds = preds.log_softmax(dim=2)
        targets = labels.to(device, dtype=torch.float32)
        input_lengths = torch.tensor(
            [len(l) for l in preds.permute(1, 0, 2)], dtype=torch.float32, device=device
        )
        target_lengths = torch.tensor(labels_length, dtype=torch.float32, device=device)

        loss = ctc_loss(log_preds, targets, input_lengths, target_lengths)
        running_loss += loss.item() * len(train_data)
        print("epoch:{}, iter:{}, loss:{}".format(epoch, i, loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm(params, max_norm=0.1)
        optimizer.step()
    epoch_loss = running_loss / len(train_dataset)
    print("epoch:{}, epoch_loss:{}".format(epoch, epoch_loss))
