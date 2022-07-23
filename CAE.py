import torch.nn as nn
import torch
import time
from preprocess import *
from evaluation import *

class CAE(nn.Module):
    def __init__(self, device):
        super(CAE, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=int(3//2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=int(3//2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.flat = nn.Flatten()
        self.deconv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=int(3//2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=int(3//2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=int(3//2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=int(3//2)),
            nn.Sigmoid()
        )

    def forward(self, x):  # (batch_size, 1, 33, 33), (batch_size, 1, 27, 27)
        z = torch.zeros(x.shape[0], x.shape[1], x.shape[2]+1, x.shape[3]+1).to(self.device)
        z[:,:,0:z.shape[2]-1, 0:z.shape[3]-1] = x  # (batch_size, 16, 34, 34), (batch_size, 1, 28, 28)
        x = self.conv1(z)  # (batch_size, 16, 16, 16), (batch_size, 16, 14, 14)
        x = self.conv2(x)  # (batch_size, 32, 8, 8), (batch_size, 32, 7, 7)
        x = self.conv3(x)  # (batch_size, 64, 4, 4), (batch_size, 64, 3, 3)
        x = self.flat(x)  # (batch_size, 1024), (batch_size, 576)
        flat_size = x.shape[1]
        x = nn.Linear(x.shape[1], 10).to(self.device)(x)  # (batch_size, 10), (batch_size, 10)
        x = nn.Linear(10, flat_size).to(self.device)(x)  # (batch_size, 1024), (batch_size, 576)
        x = x.contiguous().view(x.shape[0], 64, int(((x.shape[1]/64)**0.5)), int(((x.shape[1]/64)**0.5)))  # (batch_size, 64, 4, 4), (batch_size, 64, 3, 3)
        x = self.deconv3(x)  # (batch_size, 64, 8, 8), (batch_size, 64, 6, 6)
        x = self.deconv2(x)  # (batch_size, 32, 16, 16), (batch_size, 32, 12, 12)
        x = self.deconv1(x)  # (batch_size, 16, 32, 32), (batch_size, 16, 24, 24)
        # z = torch.zeros(x.shape[0], x.shape[1], x.shape[2]+1, x.shape[3]+1).to(self.device)
        # z[:, :, 0:z.shape[2]-1, 0:z.shape[3]-1] = x  # (batch_size, 16, 33, 33)
        z = torch.zeros(x.shape[0], x.shape[1], x.shape[2]+3, x.shape[3]+3).to(self.device)
        z[:, :, 1:z.shape[2]-2, 1:z.shape[3]-2] = x  # (batch_size, 16, 27, 27)
        x = self.deconv4(z)  # (batch_size, 1, 33, 33), (batch_size, 16, 27, 27)
        return x

def training(model, loader, epochs, optimizer, criterion, device):
    print(f"Training model for {epochs} epochs..")
    train_start = time.time()
    train_loss = []
    model.to(device)
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        batch_loss = []
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            reconstruction = model(x)
            loss = torch.sqrt(criterion(x, reconstruction))
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss = np.array(batch_loss).mean()
        train_loss.append(epoch_loss)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"[Epoch {epoch + 1}]: train_loss={epoch_loss: .5f}  epoch_time={epoch_time: .5f} s")
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), f"weights/CAE/cae{epoch+1}.pt")
    train_end = time.time()
    train_time = train_end - train_start
    print(f"Training done in {train_time: .5f}s..")
    plt.plot(train_loss, color="blue", label="cae_train_losses")
    plt.legend()
    plt.savefig("cae_train_losses_swat.png")
