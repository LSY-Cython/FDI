from LSTM_ED import *
import numpy as np
from evaluation import *

def validation_tep(model, weight, loader, device, percent):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    error_vecs = list()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)  # (1, win_size, n_features)
            reconstruction = model(x).numpy()  # (1, win_size, n_features)
            for i in range(reconstruction.shape[1]):
                rec_vec = reconstruction[0][i]  # (n_features, )
                raw_vec = x.numpy()[0][i]
                err_vec = np.abs(rec_vec-raw_vec)
                error_vecs.append(err_vec)
    anomalyScores, varErrors, avrErrVec, covErrInv = mahal_score(np.array(error_vecs))
    recon_limit = np.percentile(anomalyScores, percent)
    var_limit = np.percentile(varErrors, percent, axis=0)
    return recon_limit, var_limit, avrErrVec, covErrInv

def testing_tep(model, weight, loader, device, recon_limit, var_limit, avrErrVec, conErrVec):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    results = list()
    with torch.no_grad():
        i = 0
        for x in loader:
            x = x.to(device)
            reconstruction = model(x).numpy()   # (1, win_size, n_features)
            win_size = reconstruction.shape[1]
            for j in range(win_size):
                rec_vec = reconstruction[0][j]  # (n_features, )
                raw_vec = x.numpy()[0][j]
                err_vec = np.abs(rec_vec - raw_vec)
                rec_error = np.dot(np.dot(err_vec-avrErrVec, conErrVec), (err_vec-avrErrVec).T)
                var_error = np.dot(err_vec-avrErrVec, conErrVec)*(err_vec-avrErrVec)
                if rec_error<=recon_limit:
                    results.append(["normal", win_size*i+j])
                else:
                    anomalyId = np.where(var_error>var_limit)[0].tolist()
                    results.append(["anomaly", win_size*i+j, anomalyId])
            i += 1
    return results

def evaluate_tep():
    lstm_ed = LSTM_ED(input_size=33, embded_size=32, hidden_size=32, dropout=0.2)
    optimizer = torch.optim.Adam(lstm_ed.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainScaled, trainScaler = standarlize_tep_data("dataset/TEP_data/d00_te.dat")
    trainSet = SlidingWindowDataset(data=trainScaled, window_size=30, step_size=10)
    trainLoader = DataLoader(dataset=trainSet, batch_size=16, shuffle=True)

    # training(model=lstm_ed,
    #          loader=trainLoader,
    #          epochs=1000,
    #          optimizer=optimizer,
    #          criterion=criterion,
    #          device=device)

    weight = "weights/LSTM-ED/TEP/lstm_ed1000.pt"
    # 验证集估计控制限阈值
    validData = read_tep_data("dataset/TEP_data/d00.dat")
    validScaled = trainScaler.transform(validData)
    validSet = SlidingWindowDataset(data=validScaled, window_size=30, step_size=30)
    validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
    recon_limit, var_limit, avrErrVec, covErrInv = validation_tep(model=lstm_ed,
                                                                  weight=weight,
                                                                  loader=validLoader,
                                                                  device=device,
                                                                  percent=90)
    print("重构误差控制限：", recon_limit)
    print("变量误差控制限：", var_limit)

    # 批测21种TEP故障类型
    import json
    import collections
    batch_results = dict()
    FDRs, FARs, F1s = 0, 0, 0
    for id in range(1, 22, 1):
        if id < 10:
            id = f"0{id}"
        testFile = f"dataset/TEP_data/d{id}_te.dat"
        testData = read_tep_data(testFile)
        testScaled = trainScaler.transform(testData)
        testSet = SlidingWindowDataset(data=testScaled, window_size=30, step_size=30)
        testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
        test_results = testing_tep(model=lstm_ed,
                                   weight=weight,
                                   loader=testLoader,
                                   device=device,
                                   recon_limit=recon_limit,
                                   var_limit=var_limit,
                                   avrErrVec=avrErrVec,
                                   conErrVec=covErrInv)
        TP, TN, FP, FN = 0, 0, 0, 0
        rcs = list()
        for c in test_results:  # 13,14,15为过渡样本
            if c[0] == "normal" and c[1] < 160:
                TP += 1
            elif c[0] == "normal" and c[1] >= 160:
                FP += 1
            elif c[0] == "anomaly" and c[1] < 160:
                FN += 1
                rcs.extend(c[2])
            elif c[0] == "anomaly" and c[1] >= 160:
                TN += 1
                rcs.extend(c[2])
        FDR, FAR, F1 = eval_metrics(TP, TN, FP, FN)
        FDRs += FDR
        FARs += FAR
        F1s += F1
        root_causes = collections.Counter(rcs)
        batch_results[f"d{id}_te.dat"] = {
            "FDR": FDR,
            "FAR": FAR,
            "F1": F1,
            "RCs": dict(sorted(root_causes.items(), key=lambda x: x[1], reverse=True))
        }
    batch_results["aggregation"] = {
        "mFDR": FDRs / 21,
        "mFAR": FARs / 21,
        "mF1": F1s / 21,
    }
    with open("lstm-ed_tep", "w") as f:
        f.write(json.dumps(batch_results, indent=4))

def validation_swat(model, weight, loader, device, percent):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    recon_errors = list()
    var_errors = list()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)  # (1, win_size, n_features)
            reconstruction = model(x).numpy()[0]  # (win_size, n_features)
            x = x.numpy()[0]  # (win_size, n_features)
            error = np.abs(x-reconstruction)
            var_error = np.mean(error, axis=0)  # (n_features, )
            recon_error = np.sum(var_error)
            recon_errors.append(recon_error)
            var_errors.append(var_error)
    recon_limit = np.percentile(np.array(recon_errors), percent)
    var_limit = np.percentile(np.array(var_errors), percent, axis=0)
    return recon_limit, var_limit

def testing_swat(model, weight, loader, device, recon_limit, var_limit):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    results = list()
    with torch.no_grad():
        i = 0
        for x in loader:
            x = x.to(device)
            reconstruction = model(x).numpy()[0]   # (win_size, n_features)
            x = x.numpy()[0]  # (win_size, n_features)
            error = np.abs(x - reconstruction)
            var_error = np.mean(error, axis=0)  # (n_features, )
            recon_error = np.sum(var_error)
            if recon_error<=recon_limit:
                results.append(["normal", i])
            else:
                anomalyId = np.where(var_error>var_limit)[0].tolist()
                results.append(["anomaly", i, anomalyId])
            i += 1
    return results

def evaluate_swat():
    lstm_ed = LSTM_ED(input_size=27, embded_size=32, hidden_size=32, dropout=0.2)
    optimizer = torch.optim.Adam(lstm_ed.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainScaled, trainScaler = standarlize_swat_data("dataset/SWaT_data/msl/train_data.pkl")
    trainSet = SlidingWindowDataset(data=trainScaled[0:1200], window_size=30, step_size=10)
    trainLoader = DataLoader(dataset=trainSet, batch_size=16, shuffle=True)

    # training(model=lstm_ed,
    #          loader=trainLoader,
    #          epochs=1000,
    #          optimizer=optimizer,
    #          criterion=criterion,
    #          device=device)

    weight = "weights/LSTM-ED/SWaT/lstm_ed1000.pt"
    # 验证集估计控制限阈值
    validScaled = trainScaled[1200:]
    validSet = SlidingWindowDataset(data=validScaled, window_size=30, step_size=30)
    validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
    recon_limit, var_limit = validation_swat(model=lstm_ed,
                                             weight=weight,
                                             loader=validLoader,
                                             device=device,
                                             percent=100)
    print("重构误差控制限：", recon_limit)
    print("变量误差控制限：", var_limit)

    # 批测9种SWaT故障类型
    import json
    import collections
    batch_results = dict()
    FDRs, FARs, F1s = 0, 0, 0
    testScaled, testScaler = standarlize_swat_data("dataset/SWaT_data/msl/test_data.pkl")
    print("测试数据规模: ", testScaled.shape)
    testSet = SlidingWindowDataset(data=testScaled, window_size=30, step_size=30)
    testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    test_results = testing_swat(model=lstm_ed,
                                weight=weight,
                                loader=testLoader,
                                device=device,
                                recon_limit=recon_limit,
                                var_limit=var_limit)
    TP, TN, FP, FN = 0, 0, 0, 0
    rcs = list()
    for c in test_results:
        if c[0] == "normal" and (0<=c[1]<=8 or 13<=c[1]<17):
            TP += 1
        elif c[0] == "normal" and (10<=c[1]<=12 or 19<=c[1]<=67):
            FP += 1
        elif c[0] == "anomaly" and (0<=c[1]<=8 or 13<=c[1]<17):
            FN += 1
            rcs.extend(c[2])
        elif c[0] == "anomaly" and (10<=c[1]<=12 or 19<=c[1]<=67):
            TN += 1
            rcs.extend(c[2])
    FDR, FAR, F1 = eval_metrics(TP, TN, FP, FN)
    FDRs += FDR
    FARs += FAR
    F1s += F1
    root_causes = collections.Counter(rcs)
    batch_results[f"test_data"] = {
        "FDR": FDR,
        "FAR": FAR,
        "F1": F1,
        "RCs": dict(sorted(root_causes.items(), key=lambda x: x[1], reverse=True))
    }
    with open("lstm-ed_swat", "w") as f:
        f.write(json.dumps(batch_results, indent=4))

if __name__ == "__main__":
    # evaluate_tep()
    evaluate_swat()