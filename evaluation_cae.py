from CAE import *

def validation_tep(model, weight, loader, device, percent):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    recErrors = list()
    varErrors = list()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            reconstruction = model(x).squeeze().numpy()  # (n_features, n_features)
            x = x.squeeze().numpy()
            error = np.mean(np.abs(x-reconstruction), axis=1)  # (n_features, )
            varErrors.append(error)
            recErrors.append(np.mean(error))
    recon_limit = np.percentile(np.array(recErrors), percent)
    var_limit = np.percentile(np.array(varErrors), percent, axis=0)
    return recon_limit, var_limit

def testing_tep(model, weight, loader, device, recon_limit, var_limit):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    results = list()
    with torch.no_grad():
        i = 0
        for x in loader:
            x = x.to(device)
            reconstruction = model(x).squeeze().numpy()  # (n_features, n_features)
            x = x.squeeze().numpy()
            var_error = np.mean(np.abs(x-reconstruction), axis=1)  # (n_features, )
            recon_error = np.mean(var_error)
            if recon_error<=recon_limit:
                results.append(["normal", i])
            else:
                anomalyId = np.where(var_error>var_limit)[0].tolist()
                results.append(["anomaly", i, anomalyId])
            i+=1
    return results

def evaluation_tep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cae = CAE(device)
    optimizer = torch.optim.Adam(cae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    trainScaled, trainScaler = standarlize_tep_data("dataset/TEP_data/d00_te.dat")
    trainSet = CorrelationImagesDataset(data=trainScaled, window_size=30, step_size=10)
    trainLoader = DataLoader(dataset=trainSet, batch_size=16, shuffle=True)

    # training(model=cae,
    #          loader=trainLoader,
    #          epochs=1000,
    #          optimizer=optimizer,
    #          criterion=criterion,
    #          device=device)

    weight = "weights/CAE/TEP/cae500.pt"
    # 验证集估计控制限阈值
    validData = read_tep_data("dataset/TEP_data/d00.dat")
    validScaled = trainScaler.transform(validData)
    validSet = CorrelationImagesDataset(data=validScaled, window_size=30, step_size=10)
    validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
    recon_limit, var_limit = validation_tep(model=cae,
                                            weight=weight,
                                            loader=validLoader,
                                            device=device,
                                            percent=96)
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
        testSet = CorrelationImagesDataset(data=testScaled, window_size=30, step_size=10)
        testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
        test_results = testing_tep(model=cae,
                                   weight=weight,
                                   loader=testLoader,
                                   device=device,
                                   recon_limit=recon_limit,
                                   var_limit=var_limit)
        TP, TN, FP, FN = 0, 0, 0, 0
        rcs = list()
        for c in test_results:  # 5为过渡样本
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
    with open("cae_tep", "w") as f:
        f.write(json.dumps(batch_results, indent=4))

def validation_swat(model, weight, loader, device, percent):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    recErrors = list()
    varErrors = list()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            reconstruction = model(x).squeeze().numpy()  # (n_features, n_features)
            x = x.squeeze().numpy()
            error = np.mean(np.abs(x-reconstruction), axis=1)  # (n_features, )
            varErrors.append(error)
            recErrors.append(np.sum(error))
    recon_limit = np.percentile(np.array(recErrors), percent)
    var_limit = np.percentile(np.array(varErrors), percent, axis=0)
    return recon_limit, var_limit

def testing_swat(model, weight, loader, device, recon_limit, var_limit):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    results = list()
    with torch.no_grad():
        i = 0
        for x in loader:
            x = x.to(device)
            reconstruction = model(x).squeeze().numpy()  # (n_features, n_features)
            x = x.squeeze().numpy()
            var_error = np.mean(np.abs(x-reconstruction), axis=1)  # (n_features, )
            recon_error = np.sum(var_error)
            if recon_error<=recon_limit:
                results.append(["normal", i])
            else:
                anomalyId = np.where(var_error>var_limit)[0].tolist()
                results.append(["anomaly", i, anomalyId])
            i+=1
    return results

def evaluation_swat():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cae = CAE(device)
    optimizer = torch.optim.Adam(cae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    trainScaled, trainScaler = standarlize_swat_data("dataset/SWaT_data/msl/train_data.pkl")
    trainSet = CorrelationImagesDataset(data=trainScaled[0:1200], window_size=30, step_size=10)
    trainLoader = DataLoader(dataset=trainSet, batch_size=16, shuffle=True)

    # training(model=cae,
    #          loader=trainLoader,
    #          epochs=1000,
    #          optimizer=optimizer,
    #          criterion=criterion,
    #          device=device)

    weight = "weights/CAE/SWaT/cae1000.pt"
    # 验证集估计控制限阈值
    validScaled = trainScaled[1200:]
    validSet = CorrelationImagesDataset(data=validScaled, window_size=30, step_size=30)
    validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
    recon_limit, var_limit = validation_swat(model=cae,
                                             weight=weight,
                                             loader=validLoader,
                                             device=device,
                                             percent=75)
    print("重构误差控制限：", recon_limit)
    print("变量误差控制限：", var_limit)

    # 批测9种SWaT故障类型
    import json
    import collections
    batch_results = dict()
    FDRs, FARs, F1s = 0, 0, 0
    testScaled, testScaler = standarlize_swat_data("dataset/SWaT_data/msl/test_data.pkl")
    print("测试数据规模: ", testScaled.shape)
    testSet = CorrelationImagesDataset(data=testScaled, window_size=30, step_size=30)
    testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    test_results = testing_swat(model=cae,
                                weight=weight,
                                loader=testLoader,
                                device=device,
                                recon_limit=recon_limit,
                                var_limit=var_limit)
    print(test_results)
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
    with open("cae_swat", "w") as f:
        f.write(json.dumps(batch_results, indent=4))

if __name__ == "__main__":
    # evaluation_tep()
    evaluation_swat()