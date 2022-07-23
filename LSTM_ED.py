import torch.nn as nn
import torch
import time
from preprocess import *

class LSTM_ED(nn.Module):
    def __init__(self, input_size, embded_size, hidden_size, dropout):
        super(LSTM_ED, self).__init__()
        self.input_size = input_size
        self.embded_size = embded_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.encoder = nn.Linear(self.input_size, self.embded_size)
        self.rnn = nn.LSTM(input_size=self.embded_size, hidden_size=self.hidden_size, batch_first=True)
        self.decoder = nn.Linear(self.hidden_size, self.input_size)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):  # (batch_size, win_size, n_features)
        enc = self.encoder(x.view(x.shape[0]*x.shape[1], x.shape[2]))  # (batch_size*win_size, embded_size)
        drop = self.drop(enc.view(x.shape[0], x.shape[1], enc.shape[1]))  # (batch_size, win_size, embded_size)
        output, _ = self.rnn(drop)  # (batch_size, win_size, hidden_size)
        drop = self.drop(output)  # (batch_size, win_size, hidden_size)
        dec = self.decoder(drop.contiguous().view(drop.shape[0]*drop.shape[1], drop.shape[2]))  # (batch_size*win_size, input_size)
        rec = dec.view(x.shape[0], x.shape[1], x.shape[2])  # (batch_size, win_size, input_size)
        return rec

# 计算误差向量组间基于马氏距离的异常评分
def mahal_score(vecs):  # (nums, dims)
    cov = np.cov(vecs.T)  # 协方差矩阵
    covInv = np.linalg.inv(cov)
    avrVec = np.mean(vecs, axis=0)  # 误差均值向量
    anomalyScores = list()
    varErrors = list()
    for v in vecs:
        a = np.dot(np.dot(v-avrVec, covInv), (v-avrVec).T)
        anomalyScores.append(a)
        varErrors.append(np.dot(v-avrVec, covInv)*(v-avrVec))  # 投影分解
    return np.array(anomalyScores), np.array(varErrors), avrVec, covInv

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
            torch.save(model.state_dict(), f"weights/LSTM-ED/lstm_ed{epoch+1}.pt")
    train_end = time.time()
    train_time = train_end - train_start
    print(f"Training done in {train_time: .5f}s..")
    plt.plot(train_loss, color="blue", label="lstm_ed_train_losses")
    plt.legend()
    plt.savefig("lstm_ed_train_losses.png")
