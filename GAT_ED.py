import torch.nn as nn
import torch
import time
from scipy.stats import norm
from preprocess import *
from evaluation import *

class GAT_ED(nn.Module):
    def __init__(self, input_size, win_size, control_size, kernel_size, hidden_size, embed_size, dropout):
        super(GAT_ED, self).__init__()
        self.input_size = input_size  # 传感器数量
        self.win_size = win_size
        self.control_size = control_size  # 执行器数量
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        # 一维卷积滤波
        self.cnn = nn.Sequential(
            nn.ConstantPad1d((kernel_size-1)//2, 0.0),  # 左右各插补(kernel_size-1)//2个0
            nn.Conv1d(in_channels=self.input_size, out_channels=self.input_size, kernel_size=self.kernel_size),
            nn.SELU()
        )
        # 时序依赖性编码
        self.rnn_enc = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        # 多模过程图注意力
        self.mp_gat = MP_GAT(node_num=self.input_size, seq_size=self.win_size, embed_size=self.embed_size)
        # 闭环控制图注意力
        self.fc_gat = FC_GAT(node_num=self.input_size, neigh_num=self.control_size, seq_size=self.win_size, embed_size=self.embed_size)
        # 时空相关性融合编码
        # self.encoder = nn.GRU(input_size=self.hidden_size+2*self.input_size, hidden_size=self.hidden_size, batch_first=True)
        # self.encoder = nn.GRU(input_size=2*self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.encoder = nn.GRU(input_size=self.hidden_size+self.input_size, hidden_size=self.hidden_size, batch_first=True)
        # 重构解码器
        self.rnn_dec = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.fc_dec = nn.Linear(self.hidden_size, self.input_size)
        # 预测解码器
        self.fc_pred = nn.Sequential(
            nn.Linear(self.hidden_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.input_size)
        )
    def forward(self, x, y):  # (batch_size, seq_size, n_features), (batch_size, seq_size, n_controls)
        c = (self.cnn(x.permute(0, 2, 1))).permute(0, 2, 1)  # (batch_size, seq_size, n_features)
        h_out, _ = self.rnn_enc(c)  # (batch_size, seq_size, hidden_size)
        h_mp, a_mp = self.mp_gat(c)  # (batch_size, seq_size, n_features), (batch_size, n_features, n_features)
        # h_fc, a_fc = self.fc_gat(c, y)  # (batch_size, seq_size, n_features), (batch_size, n_features, n_controls+1)
        # h_cat = torch.cat((h_out, h_mp, h_fc), dim=2)  # (batch_size, seq_size, hidden_size+2*n_features)
        # h_cat = torch.cat((h_mp, h_fc), dim=2)  # (batch_size, seq_size, 2*n_features)
        h_cat = torch.cat((h_out, h_mp), dim=2)  # (batch_size, seq_size, hidden_size+n_features)
        h_enc, _ = self.encoder(h_cat)  # (batch_size, seq_size, hidden_size)
        h_enc = self.drop(h_enc)
        h_dec, _ = self.rnn_dec(h_enc)  # (batch_size, seq_size, hidden_size)
        recon = self.fc_dec(h_dec)  # (batch_size, seq_size, n_features)
        pred = self.fc_pred(h_dec[:, 0, :]).squeeze(1)  # (batch_size, n_features)
        return recon, pred

class MP_GAT(nn.Module):
    def __init__(self, node_num, seq_size, embed_size):
        super(MP_GAT, self).__init__()
        self.node_num = node_num
        self.seq_size = seq_size
        self.embed_size = embed_size
        self.lin_w = nn.Linear(2*self.seq_size, self.embed_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.lin_a = nn.Linear(self.embed_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)
    def forward(self, x):  # (batch_size, seq_size, n_features)
        x_cat = self.concatenate(x)  # (batch_size, n_features, n_features, seq_size*2)
        x_w = self.leaky_relu(self.lin_w(x_cat))  # (batch_size, n_features, n_features, embed_size)
        x_a = self.lin_a(x_w)  # (batch_size, n_features, n_features, 1)
        e = x_a.squeeze(dim=3)  # (batch_size, n_features, n_features)
        attention = torch.softmax(e, dim=2)  # (batch_size, n_features, n_features)
        attention = self.drop(attention)
        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x.permute(0, 2, 1)))  # (batch_size, n_features, seq_size)
        return h.permute(0, 2, 1), attention  # (batch_size, seq_size, n_features)

    def concatenate(self, v):  # (batch_size, win_size, n_features)
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """
        v = v.permute(0, 2, 1)  # (batch_size, n_features, seq_size)
        vi = v.repeat_interleave(self.node_num, dim=1)  # (batch_size, n_features*n_features, seq_size)
        vj = v.repeat(1, self.node_num, 1)  # (batch_size, n_features*n_features, seq_size)
        v_cat = torch.cat((vi, vj), dim=2)  # (batch_size, n_features*n_features, seq_size*2)
        return v_cat.view(v.size(0), self.node_num, self.node_num, 2*v.shape[2])  # (batch_size, n_features, n_features, seq_size*2)

class FC_GAT(nn.Module):
    def __init__(self, node_num, neigh_num, seq_size, embed_size):
        super(FC_GAT, self).__init__()
        self.node_num = node_num
        self.neigh_num = neigh_num
        self.seq_size = seq_size
        self.embed_size = embed_size
        self.lin_w = nn.Linear(2 * self.seq_size, self.embed_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.lin_a = nn.Linear(self.embed_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)
    def forward(self, x, y):  # (batch_size, seq_size, n_features), (batch_size, seq_size, n_controls)
        x_cat, vj = self.concatenate(x, y)  # (batch_size, n_features, n_controls+1, seq_size*2)
        x_w = self.leaky_relu(self.lin_w(x_cat))  # (batch_size, n_features, n_controls+1, embed_size)
        x_a = self.lin_a(x_w)  # (batch_size, n_features, n_controls+1, 1)
        e = x_a.squeeze(dim=3)  # (batch_size, n_features, n_controls+1)
        attention = torch.softmax(e, dim=2)  # (batch_size, n_features, n_controls+1)
        attention = self.drop(attention)
        # Computing new node features using the attention
        h = torch.empty_like(x.permute(0, 2, 1))  # (batch_size, n_features, seq_size)
        for i in range(self.node_num):
            # (batch_size,1,n_controls+1)*(batch_size,n_controls+1,seq_size)=(batch_size,1,seq_size)
            hi = self.sigmoid(torch.matmul(attention[:,i,:].unsqueeze(1), vj[:,i*(self.neigh_num+1):(i+1)*(self.neigh_num+1),:]))
            h[:, i, :] = hi.squeeze(1)
        return h.permute(0, 2, 1), attention  # (batch_size, seq_size, n_features)
    def concatenate(self, v, c):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            x1 || x1
            x1 || y1,
            ...
            x1 || ym,
            x2 || x2,
            x2 || y1,
            ...
            x2 || ym,
            ...
            ...
            xK || xK,
            xK || y1,
            ...
            xK || ym,
        """
        v = v.permute(0, 2, 1)  # (batch_size, n_features, seq_size)
        c = c.permute(0, 2, 1)  # (batch_size, n_controls, seq_size)
        vi = v.repeat_interleave(self.neigh_num+1, dim=1)  # (batch_size, n_features*(n_controls+1), seq_size)
        vj = torch.empty_like(vi)  # (batch_size, n_features*(n_controls+1), seq_size)
        for i in range(self.node_num):
            vj[:, i*(self.neigh_num+1), :] = v[:, i, :]
            vj[:, i*(self.neigh_num+1)+1:(i+1)*(self.neigh_num+1), :] = c
        v_cat = torch.cat((vi, vj), dim=2)  # (batch_size, n_features*(n_controls+1), seq_size*2)
        return v_cat.view(v.size(0),self.node_num,self.neigh_num+1,2*v.shape[2]),vj  # (batch_size,n_features,n_controls+1,seq_size*2)

def training(model, loader, epochs, optimizer, criterion, device, alpha):
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
            # p = x[:, :, 0:22]  # 过程数据
            # c = x[:, :, 22:33]  # 控制数据
            # l = y[:, 0:22]
            p = x[:, :, swatSensorId]  # 过程数据
            c = x[:, :, swatActuatorId]  # 控制数据
            l = y[:, swatSensorId]
            optimizer.zero_grad()
            reconstruction, prediction = model(p, c)
            loss = alpha*torch.sqrt(criterion(p,reconstruction)) + (1-alpha)*torch.sqrt(criterion(l, prediction))
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss = np.array(batch_loss).mean()
        train_loss.append(epoch_loss)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"[Epoch {epoch + 1}]: train_loss={epoch_loss: .5f}  epoch_time={epoch_time: .5f} s")
        if (epoch+1) % 25 == 0:
            torch.save(model.state_dict(), f"weights/GAT-ED/gat_ed{epoch+1}.pt")
    train_end = time.time()
    train_time = train_end - train_start
    print(f"Training done in {train_time: .5f}s..")
    plt.plot(train_loss, color="blue", label="gat_ed_train_losses")
    plt.legend()
    plt.savefig("gat_ed_train_losses_swat.png")


