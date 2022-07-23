from preprocess import *
from VCkNN import *
from evaluation import *

def evaluate_tep():
    trainScaled, trainScaler = standarlize_tep_data("dataset/TEP_data/d00_te.dat")
    vcknn = VCkNN(k=50,
                  alpha=0.01,
                  trainData=trainScaled,
                  testData=None
                  )
    vcknn.run_vcknn()
    print("vcknn距离控制限：", vcknn.dist_limit)
    print("vcknn变量贡献度控制限：", vcknn.vc_limit)
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
        vcknn.testData = testScaled
        vcknn.test_vcknn()
        test_results = vcknn.test_results
        TP, TN, FP, FN = 0, 0, 0, 0
        rcs = list()
        for c in test_results:
            if c[0] == "normal" and c[1] < 160:
                TP += 1
            elif c[0] == "normal" and c[1] >= 160:
                FP += 1
            elif c[0] == "anomaly" and c[1] <= 160:
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
            "mFDR": FDR,
            "mFAR": FAR,
            "mF1": F1,
            "mRCs": dict(sorted(root_causes.items(), key=lambda x: x[1], reverse=True))
        }
    batch_results["aggregation"] = {
        "FDR": FDRs / 21,
        "FAR": FARs / 21,
        "F1": F1s / 21,
    }
    with open("vcknn_tep", "w") as f:
        f.write(json.dumps(batch_results, indent=4))

def evaluate_swat():
    trainScaled, trainScaler = standarlize_swat_data("dataset/SWaT_data/msl/train_data.pkl")
    print("训练数据规模：", trainScaled.shape)
    vcknn = VCkNN(k=50,
                  alpha=0.01,
                  trainData=trainScaled,
                  testData=None
                  )
    vcknn.run_vcknn()
    print("vcknn距离控制限：", vcknn.dist_limit)
    print("vcknn变量贡献度控制限：", vcknn.vc_limit)

    import json
    import collections
    batch_results = dict()
    FDRs, FARs, F1s = 0, 0, 0
    testScaled, testScaler = standarlize_swat_data("dataset/SWaT_data/msl/test_data.pkl")
    print("测试数据规模: ", testScaled.shape)
    vcknn.testData = testScaled
    vcknn.test_vcknn()
    test_results = vcknn.test_results
    TP, TN, FP, FN = 0, 0, 0, 0
    rcs = list()
    for c in test_results:
        if c[0] == "normal" and (0<=c[1]<290 or 391<=c[1]<550):
            TP += 1
        elif c[0] == "normal" and (290<=c[1]<391 or 550<=c[1]<2049):
            FP += 1
        elif c[0] == "anomaly" and (0<=c[1]<290 or 391<=c[1]<550):
            FN += 1
            rcs.extend(c[2])
        elif c[0] == "anomaly" and (290<=c[1]<391 or 550<=c[1]<2049):
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
    with open("vcknn_swat", "w") as f:
        f.write(json.dumps(batch_results, indent=4))

if __name__ == "__main__":
    # evaluate_tep()
    evaluate_swat()