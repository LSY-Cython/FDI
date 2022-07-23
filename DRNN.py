import torch.nn as nn
import torch
import time
from preprocess import *

class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.lin = nn.Sequential(
            nn.Linear(self.hidden_size, self.input_size),
            nn.Sigmoid()
        )
    def forward(self, x):  # (batch_size, win_size, n_features)
        output, hn = self.rnn(x)  # (1, batch_size, n_features)
        x = self.lin(hn.squeeze())  # (batch_size, n_features)
        return x

# 计算预测序列协方差矩阵的L1范数
def covariance_regular(pred_mat, alpha):  # (batch_size, n_features)
    N = pred_mat.shape[0]
    S = pred_mat.shape[-1]
    pred_mat = pred_mat.view(N, -1)
    mean = torch.mean(pred_mat, dim=0).unsqueeze(0)  # (1, n_features)
    X = pred_mat - mean
    cov = X.transpose(1, 0) @ X / (N-1)
    lc = torch.sum(cov) / S
    return alpha*lc

def training(model, loader, epochs, optimizer, criterion, device):
    print(f"Training model for {epochs} epochs..")
    train_start = time.time()
    train_loss = []
    model.to(device)
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        batch_loss = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = torch.sqrt(criterion(y, prediction)) + covariance_regular(prediction, 0.01)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss = np.array(batch_loss).mean()
        train_loss.append(epoch_loss)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"[Epoch {epoch + 1}]: train_loss={epoch_loss: .5f}  epoch_time={epoch_time: .5f} s")
        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), f"weights/DRNN/drnn{epoch+1}.pt")
    train_end = time.time()
    train_time = train_end - train_start
    print(f"Training done in {train_time: .5f}s..")
    plt.plot(train_loss, color="blue", label="drnn_train_loss")
    plt.legend()
    plt.savefig("drnn_train_losses.png")