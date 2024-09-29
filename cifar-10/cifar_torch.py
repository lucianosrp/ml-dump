import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm, trange


class ModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = self._make_layer(3, 64)
        self.fc2 = self._make_layer(64, 128)
        self.fc3 = self._make_layer(128, 256)
        self.fc4 = self._make_layer(256, 512)

        self.fc5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    def _make_layer(self, in_features: int, out_features: int):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


def train_epoch(device, epoch, model, data_loader, optimizer, loss_fn):
    model = model.to(device)
    model.train()
    loss_epoch = 0
    acc_epoch = 0
    for batch_idx, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss_batch = loss.item()
        loss_epoch += loss_batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc_batch = accuracy(logits, y)
        acc_epoch += acc_batch
        if batch_idx % 10 == 0:
            tqdm.write(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}".format(
                    epoch,
                    batch_idx * len(X),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.item(),
                    acc_batch,
                )
            )

    return loss_epoch / len(data_loader), acc_epoch / len(data_loader)


def test_epoch(device, epoch, model, data_loader, optimizer, loss_fn):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        loss_epoch = 0
        acc_epoch = 0
        for batch_idx, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss_epoch += loss.item()
            acc_epoch += accuracy(logits, y)

        tqdm.write(
            "Test Epoch: {} \tLoss: {:.6f}\tAcc: {:.6f}".format(
                epoch, loss_epoch / len(data_loader), acc_epoch / len(data_loader)
            )
        )

        return loss_epoch / len(data_loader), acc_epoch / len(data_loader)


def accuracy(preds, labels):
    predicted_classes = torch.argmax(preds, dim=1)
    correct = (predicted_classes == labels).float()
    acc = correct.sum() / len(correct)
    return acc


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    data_train = CIFAR10(root="./data", train=True, download=True, transform=transform)
    data_test = CIFAR10(root="./data", train=False, download=True, transform=transform)

    model = ModelV0()
    data_loader_train = DataLoader(
        data_train, batch_size=32, num_workers=2, shuffle=True
    )

    data_loader_test = DataLoader(data_test, batch_size=32, num_workers=2, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

    for e in trange(20):
        tr_loss, tr_acc = train_epoch(
            device, e, model, data_loader_train, optimizer, loss_fn
        )
        tt_loss, tt_acc = test_epoch(
            device, e, model, data_loader_test, optimizer, loss_fn
        )
        scheduler.step()

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(tt_loss)
        test_acc.append(tt_acc)
