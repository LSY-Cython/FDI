from GAT_ED import *

def validation_tep(model, weight, loader, device, alpha, confidence):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    varErrors = list()
    jointErrors = list()
    varParams = list()
    var_limit = list()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = x[:, :, 0:22]  # 过程数据
            c = x[:, :, 22:33]  # 控制数据
            l = y[:, 0:22]
            reconstruction, prediction = model(p, c)  # (1, seq_size, n_features), (1, n_features)
            reconstruction = reconstruction.squeeze().to("cpu").numpy()  # (seq_size, n_features)
            prediction = prediction.squeeze().to("cpu").numpy()  # (n_features, )
            reconDiff = np.abs(p.squeeze().to("cpu").numpy()-reconstruction)  # (seq_size, n_features)
            reconVarError = np.mean(reconDiff, axis=0)  # (n_features, )
            predVarError = np.abs(l.squeeze().to("cpu").numpy()-prediction)  # (n_features, )
            varError = alpha*reconVarError + (1-alpha)*predVarError  # (n_features, )
            jointError = np.sum(varError)
            varErrors.append(varError)
            jointErrors.append(jointError)
    varErrors = np.array(varErrors)
    jointErrors = np.array(jointErrors)
    # 极大似然估计
    for i in range(varErrors.shape[1]):
        ui, oi = norm.fit(varErrors[:, i])  # 正态分布参数
        varParams.append([ui, oi])
        lower, upper = norm.interval(confidence, loc=ui, scale=oi)  # 置信区间上下临界值
        var_limit.append(upper)
    u, o = norm.fit(jointErrors)
    jointParams = [u, o]
    lower, upper = norm.interval(confidence, loc=u, scale=o)
    joint_limit = upper
    # print("变量误差分布参数：", varParams)
    # print("联合误差分布参数: ", jointParams)
    print("变量误差控制限：", var_limit)
    print("联合误差控制限：", joint_limit)
    return joint_limit, np.array(var_limit)

def testing_tep(model, weight, loader, device, alpha, joint_limit, var_limit):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    results = list()
    with torch.no_grad():
        i = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = x[:, :, 0:22]  # 过程数据
            c = x[:, :, 22:33]  # 控制数据
            l = y[:, 0:22]
            reconstruction, prediction = model(p, c)  # (1, seq_size, n_features), (1, n_features)
            reconstruction = reconstruction.squeeze().to("cpu").numpy()  # (seq_size, n_features)
            prediction = prediction.squeeze().to("cpu").numpy()  # (n_features, )
            recon_diff = np.abs(p.squeeze().to("cpu").numpy() - reconstruction)  # (seq_size, n_features)
            reconVarError = np.mean(recon_diff, axis=0)  # (n_features, )
            predVarError = np.abs(l.squeeze().to("cpu").numpy()-prediction)  # (n_features, )
            var_error = alpha*reconVarError+(1-alpha)*predVarError  # (n_features, )
            joint_error = np.sum(var_error)
            if joint_error<=joint_limit:
                results.append(["normal", i])
            else:
                anomalyId = np.where(var_error>var_limit)[0].tolist()
                results.append(["anomaly", i, anomalyId])
            i+=1
    return results

def evaluation_tep(alpha):
    gat_ed = GAT_ED(input_size=22,
                    win_size=30,
                    control_size=11,
                    kernel_size=7,
                    hidden_size=30,
                    embed_size=30,
                    dropout=0.0)
    optimizer = torch.optim.Adam(gat_ed.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainScaled, trainScaler = standarlize_tep_data("dataset/TEP_data/d00_te.dat")
    trainSet = ParallelDataset(data=trainScaled, window_size=30, step_size=10)
    trainLoader = DataLoader(dataset=trainSet, batch_size=16, shuffle=True)

    training(model=gat_ed,
             loader=trainLoader,
             epochs=1000,
             optimizer=optimizer,
             criterion=criterion,
             device=device,
             alpha=alpha)

    # epoch = 1000
    # weight = f"weights/GAT-ED/{alpha}/TEP/gat_ed{epoch}.pt"
    # # 验证集估计控制限阈值
    # validData = read_tep_data("dataset/TEP_data/d00.dat")
    # validScaled = trainScaler.transform(validData)
    # validSet = ParallelDataset(data=validScaled, window_size=30, step_size=10)
    # validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
    # joint_limit, var_limit = validation_tep(model=gat_ed,
    #                                         weight=weight,
    #                                         loader=validLoader,
    #                                         device=device,
    #                                         alpha=alpha,
    #                                         confidence=0.95)
    # # 批测21种TEP故障类型
    # import json
    # import collections
    # batch_results = dict()
    # FDRs, FARs, F1s = 0, 0, 0
    # for id in range(1, 22, 1):
    #     if id < 10:
    #         id = f"0{id}"
    #     testFile = f"dataset/TEP_data/d{id}_te.dat"
    #     testData = read_tep_data(testFile)
    #     testScaled = trainScaler.transform(testData)
    #     testSet = ParallelDataset(data=testScaled, window_size=30, step_size=10)
    #     testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    #     test_results = testing_tep(model=gat_ed,
    #                                weight=weight,
    #                                loader=testLoader,
    #                                device=device,
    #                                alpha=alpha,
    #                                joint_limit=joint_limit,
    #                                var_limit=var_limit)
    #     TP, TN, FP, FN = 0, 0, 0, 0
    #     rcs = list()
    #     for c in test_results:  # 5为过渡样本
    #         if c[0] == "normal" and c[1] < 13:
    #             TP += 1
    #         elif c[0] == "normal" and c[1] >= 16:
    #             FP += 1
    #         elif c[0] == "anomaly" and c[1] < 13:
    #             FN += 1
    #             rcs.extend(c[2])
    #         elif c[0] == "anomaly" and c[1] >= 16:
    #             TN += 1
    #             rcs.extend(c[2])
    #     FDR, FAR, F1 = eval_metrics(TP, TN, FP, FN)
    #     FDRs += FDR
    #     FARs += FAR
    #     F1s += F1
    #     root_causes = collections.Counter(rcs)
    #     batch_results[f"d{id}_te.dat"] = {
    #         "FDR": FDR,
    #         "FAR": FAR,
    #         "F1": F1,
    #         "RCs": dict(sorted(root_causes.items(), key=lambda x: x[1], reverse=True))
    #     }
    # batch_results["aggregation"] = {
    #     "mFDR": FDRs / 21,
    #     "mFAR": FARs / 21,
    #     "mF1": F1s / 21,
    # }
    # with open(f"gat_ed_tep_{alpha}", "w") as f:
    #     f.write(json.dumps(batch_results, indent=4))

def validation_swat(model, weight, loader, device, alpha, confidence):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    varErrors = list()
    jointErrors = list()
    varParams = list()
    var_limit = list()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = x[:, :, swatSensorId]  # 过程数据
            c = x[:, :, swatActuatorId]  # 控制数据
            l = y[:, swatSensorId]
            reconstruction, prediction = model(p, c)  # (1, seq_size, n_features), (1, n_features)
            reconstruction = reconstruction.squeeze().to("cpu").numpy()  # (seq_size, n_features)
            prediction = prediction.squeeze().to("cpu").numpy()  # (n_features, )
            reconDiff = np.abs(p.squeeze().to("cpu").numpy()-reconstruction)  # (seq_size, n_features)
            reconVarError = np.mean(reconDiff, axis=0)  # (n_features, )
            predVarError = np.abs(l.squeeze().to("cpu").numpy()-prediction)  # (n_features, )
            varError = alpha*reconVarError + (1-alpha)*predVarError  # (n_features, )
            jointError = np.sum(varError)
            varErrors.append(varError)
            jointErrors.append(jointError)
    varErrors = np.array(varErrors)
    jointErrors = np.array(jointErrors)
    # 极大似然估计
    for i in range(varErrors.shape[1]):
        ui, oi = norm.fit(varErrors[:, i])  # 正态分布参数
        varParams.append([ui, oi])
        lower, upper = norm.interval(confidence, loc=ui, scale=oi)  # 置信区间上下临界值
        var_limit.append(upper)
    u, o = norm.fit(jointErrors)
    jointParams = [u, o]
    lower, upper = norm.interval(confidence, loc=u, scale=o)
    joint_limit = upper
    # print("变量误差分布参数：", varParams)
    # print("联合误差分布参数: ", jointParams)
    # print("变量误差控制限：", var_limit)
    # print("联合误差控制限：", joint_limit)
    return joint_limit, np.array(var_limit)

def testing_swat(model, weight, loader, device, alpha, joint_limit, var_limit):
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    results = list()
    with torch.no_grad():
        i = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = x[:, :, swatSensorId]  # 过程数据
            c = x[:, :, swatActuatorId]  # 控制数据
            l = y[:, swatSensorId]
            reconstruction, prediction = model(p, c)  # (1, seq_size, n_features), (1, n_features)
            reconstruction = reconstruction.squeeze().to("cpu").numpy()  # (seq_size, n_features)
            prediction = prediction.squeeze().to("cpu").numpy()  # (n_features, )
            recon_diff = np.abs(p.squeeze().to("cpu").numpy() - reconstruction)  # (seq_size, n_features)
            reconVarError = np.mean(recon_diff, axis=0)  # (n_features, )
            predVarError = np.abs(l.squeeze().to("cpu").numpy()-prediction)  # (n_features, )
            var_error = alpha*reconVarError+(1-alpha)*predVarError  # (n_features, )
            joint_error = np.sum(var_error)
            if joint_error<=joint_limit:
                results.append(["normal", i])
            else:
                anomalyId = np.where(var_error>var_limit)[0].tolist()
                results.append(["anomaly", i, anomalyId])
            i+=1
    return results

def evaluation_swat(alpha):
    gat_ed = GAT_ED(input_size=21,
                    win_size=30,
                    control_size=6,
                    kernel_size=7,
                    hidden_size=30,
                    embed_size=30,
                    dropout=0.0)
    optimizer = torch.optim.Adam(gat_ed.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainScaled, trainScaler = standarlize_swat_data("dataset/SWaT_data/msl/train_data.pkl")
    trainSet = ParallelDataset(data=trainScaled[0:1200], window_size=30, step_size=10)
    trainLoader = DataLoader(dataset=trainSet, batch_size=16, shuffle=True)

    training(model=gat_ed,
             loader=trainLoader,
             epochs=1000,
             optimizer=optimizer,
             criterion=criterion,
             device=device,
             alpha=alpha)

    # epoch = 1000
    # weight = f"weights/GAT-ED/{alpha}/SWaT/gat_ed{epoch}.pt"
    # # 验证集估计控制限阈值
    # validScaled = trainScaled[1200:]
    # validSet = ParallelDataset(data=validScaled, window_size=30, step_size=30)
    # validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
    # joint_limit, var_limit = validation_swat(model=gat_ed,
    #                                          weight=weight,
    #                                          loader=validLoader,
    #                                          device=device,
    #                                          alpha=alpha,
    #                                          confidence=0.9)
    # print("联合误差控制限：", joint_limit)
    # print("变量误差控制限：", var_limit)
    #
    # # 批测9种SWaT故障类型
    # import json
    # import collections
    # batch_results = dict()
    # FDRs, FARs, F1s = 0, 0, 0
    # testScaled, testScaler = standarlize_swat_data("dataset/SWaT_data/msl/test_data.pkl")
    # print("测试数据规模: ", testScaled.shape)
    # testSet = ParallelDataset(data=testScaled, window_size=30, step_size=30)
    # testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    # test_results = testing_swat(model=gat_ed,
    #                             weight=weight,
    #                             loader=testLoader,
    #                             device=device,
    #                             alpha=alpha,
    #                             joint_limit=joint_limit,
    #                             var_limit=var_limit)
    # print(test_results)
    # TP, TN, FP, FN = 0, 0, 0, 0
    # rcs = list()
    # for c in test_results:
    #     if c[0] == "normal" and (0 <= c[1] <= 8 or 13 <= c[1] < 17):
    #         TP += 1
    #     elif c[0] == "normal" and (10 <= c[1] <= 12 or 19 <= c[1] <= 67):
    #         FP += 1
    #     elif c[0] == "anomaly" and (0 <= c[1] <= 8 or 13 <= c[1] < 17):
    #         FN += 1
    #         rcs.extend(c[2])
    #     elif c[0] == "anomaly" and (10 <= c[1] <= 12 or 19 <= c[1] <= 67):
    #         TN += 1
    #         rcs.extend(c[2])
    # FDR, FAR, F1 = eval_metrics(TP, TN, FP, FN)
    # FDRs += FDR
    # FARs += FAR
    # F1s += F1
    # root_causes = collections.Counter(rcs)
    # batch_results[f"test_data"] = {
    #     "FDR": FDR,
    #     "FAR": FAR,
    #     "F1": F1,
    #     "RCs": dict(sorted(root_causes.items(), key=lambda x: x[1], reverse=True))
    # }
    # with open(f"gat_ed_swat_{alpha}", "w") as f:
    #     f.write(json.dumps(batch_results, indent=4))

if __name__ == "__main__":
    # evaluation_tep(alpha=0.9)
    evaluation_swat(alpha=0.9)