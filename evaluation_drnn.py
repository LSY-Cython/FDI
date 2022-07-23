from DRNN import *
from evaluation import *

def validation_tep(model, weight, loader, device, percent):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    varErrors = list()
    predErrors = list()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prediction = model(x).numpy()  # (n_features, )
            y = y.squeeze().numpy()
            var_error = np.abs(prediction-y)
            varErrors.append(var_error)
            predErrors.append(np.sum(var_error))
    pred_limit = np.percentile(np.array(predErrors), percent)
    var_limit = np.percentile(np.array(varErrors), percent, axis=0)
    return pred_limit, var_limit

def testing_tep(model, weight, loader, device, pred_limit, var_limit):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    results = list()
    with torch.no_grad():
        i = 0
        for x, y in loader:
            x = x.to(device)
            prediction = model(x).numpy()  # (n_features, )
            y = y.squeeze().numpy()
            var_error = np.abs(prediction - y)
            pred_error = np.sum(var_error)
            if pred_error<=pred_limit:
                results.append(["normal", i])
            else:
                anomalyId = np.where(var_error>var_limit)[0].tolist()
                results.append(["anomaly", i, anomalyId])
            i+=1
    return results

def evaluation_tep():
    drnn = DRNN(input_size=33, hidden_size=32)
    optimizer = torch.optim.Adam(drnn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainScaled, trainScaler = standarlize_tep_data("dataset/TEP_data/d00_te.dat")
    trainSet = PredictionDataset(data=trainScaled, window_size=30, step_size=10)
    trainLoader = DataLoader(dataset=trainSet, batch_size=16, shuffle=True)

    # training(model=drnn,
    #          loader=trainLoader,
    #          epochs=1000,
    #          optimizer=optimizer,
    #          criterion=criterion,
    #          device=device)

    weight = "weights/DRNN/TEP/drnn1000.pt"
    # 验证集估计控制限阈值
    validData = read_tep_data("dataset/TEP_data/d00.dat")
    validScaled = trainScaler.transform(validData)
    validSet = PredictionDataset(data=validScaled, window_size=30, step_size=10)
    validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
    pred_limit, var_limit = validation_tep(model=drnn,
                                           weight=weight,
                                           loader=validLoader,
                                           device=device,
                                           percent=100)
    print("预测误差控制限：", pred_limit)
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
        testSet = PredictionDataset(data=testScaled, window_size=30, step_size=10)
        testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
        test_results = testing_tep(model=drnn,
                                   weight=weight,
                                   loader=testLoader,
                                   device=device,
                                   pred_limit=pred_limit,
                                   var_limit=var_limit)
        TP, TN, FP, FN = 0, 0, 0, 0
        rcs = list()
        for c in test_results:  # 13,14,15为过渡样本
            if c[0] == "normal" and c[1] < 13:
                TP += 1
            elif c[0] == "normal" and c[1] >= 16:
                FP += 1
            elif c[0] == "anomaly" and c[1] < 13:
                FN += 1
                rcs.extend(c[2])
            elif c[0] == "anomaly" and c[1] >= 16:
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
    with open("drnn_tep", "w") as f:
        f.write(json.dumps(batch_results, indent=4))

def validation_swat(model, weight, loader, device, percent):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    varErrors = list()
    predErrors = list()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prediction = model(x).numpy()  # (n_features, )
            y = y.squeeze().numpy()
            var_error = np.abs(prediction-y)
            varErrors.append(var_error)
            predErrors.append(np.sum(var_error))
    pred_limit = np.percentile(np.array(predErrors), percent)
    var_limit = np.percentile(np.array(varErrors), percent, axis=0)
    return pred_limit, var_limit

def testing_swat(model, weight, loader, device, pred_limit, var_limit):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    results = list()
    with torch.no_grad():
        i = 0
        for x, y in loader:
            x = x.to(device)
            prediction = model(x).numpy()  # (n_features, )
            y = y.squeeze().numpy()
            var_error = np.abs(prediction - y)
            pred_error = np.sum(var_error)
            if pred_error<=pred_limit:
                results.append(["normal", i])
            else:
                anomalyId = np.where(var_error>var_limit)[0].tolist()
                results.append(["anomaly", i, anomalyId])
            i+=1
    return results

def evaluation_swat():
    drnn = DRNN(input_size=27, hidden_size=32)
    optimizer = torch.optim.Adam(drnn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainScaled, trainScaler = standarlize_swat_data("dataset/SWaT_data/msl/train_data.pkl")
    trainSet = PredictionDataset(data=trainScaled[0:1200], window_size=30, step_size=10)
    trainLoader = DataLoader(dataset=trainSet, batch_size=16, shuffle=True)

    # training(model=drnn,
    #          loader=trainLoader,
    #          epochs=1000,
    #          optimizer=optimizer,
    #          criterion=criterion,
    #          device=device)

    weight = "weights/DRNN/SWaT/drnn1000.pt"
    # 验证集估计控制限阈值
    validScaled = trainScaled[1200:]
    validSet = PredictionDataset(data=validScaled, window_size=30, step_size=10)
    validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
    pred_limit, var_limit = validation_tep(model=drnn,
                                           weight=weight,
                                           loader=validLoader,
                                           device=device,
                                           percent=100)
    print("预测误差控制限：", pred_limit)
    print("变量误差控制限：", var_limit)

    # 批测9种SWaT故障类型
    import json
    import collections
    batch_results = dict()
    FDRs, FARs, F1s = 0, 0, 0
    testScaled, testScaler = standarlize_swat_data("dataset/SWaT_data/msl/test_data.pkl")
    print("测试数据规模: ", testScaled.shape)
    testSet = PredictionDataset(data=testScaled, window_size=30, step_size=30)
    testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    test_results = testing_swat(model=drnn,
                                weight=weight,
                                loader=testLoader,
                                device=device,
                                pred_limit=pred_limit,
                                var_limit=var_limit)
    TP, TN, FP, FN = 0, 0, 0, 0
    rcs = list()
    for c in test_results:
        if c[0] == "normal" and (0 <= c[1] <= 8 or 13 <= c[1] < 17):
            TP += 1
        elif c[0] == "normal" and (10 <= c[1] <= 12 or 19 <= c[1] <= 67):
            FP += 1
        elif c[0] == "anomaly" and (0 <= c[1] <= 8 or 13 <= c[1] < 17):
            FN += 1
            rcs.extend(c[2])
        elif c[0] == "anomaly" and (10 <= c[1] <= 12 or 19 <= c[1] <= 67):
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
    with open("drnn_swat", "w") as f:
        f.write(json.dumps(batch_results, indent=4))

if __name__ == "__main__":
    # evaluation_tep()
    evaluation_swat()