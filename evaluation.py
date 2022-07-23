
def eval_metrics(TP,TN,FP,FN):  # Positive: normal, Negative: anomaly
    FDR = TN/(TN+FP)  # 故障检测率
    if TN+FN != 0:
        FAR = FN/(TN+FN)  # 误报率
    else:
        FAR = 0.0
    if TP+FP == 0:
        Precision = 0.01
    else:
        Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = 2*(Precision*Recall)/(Precision+Recall)
    return FDR, FAR, F1