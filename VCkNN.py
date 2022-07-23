from sklearn.neighbors import NearestNeighbors
from preprocess import *

class VCkNN:
    def __init__(self, k, alpha, trainData, testData):
        self.k = k  # 邻居数
        self.alpha = alpha  # 容错率
        self.trainData = trainData
        self.testData = testData
        # self.percent = len(self.trainData) * (1-self.alpha)
        # if self.percent > 100:
        #     self.percent = 100
        self.percent = 100
    def train_knn_estimator(self):  # 拟合knn距离空间
        self.neigh = NearestNeighbors(n_neighbors=self.k)  # (n_samples, n_features)
        self.neigh.fit(self.trainData)
    def cal_knn_dist(self, samples, sig):  # 计算每个样本点的平均knn距离
        if sig == "train":
            dists, knn_neighbors = self.neigh.kneighbors(samples, self.k+1)  # (n_samples, k+1)
            knn_dists = np.mean(np.power(dists[:, 1:], 2), axis=1)  # (n_samples, )
            # print("训练数据knn距离：", knn_dists.tolist())
            return knn_dists, knn_neighbors[:, 1:]
        elif sig == "test":
            dists, knn_neighbors = self.neigh.kneighbors(samples, self.k)  # (n_samples, k)
            knn_dists = np.mean(np.power(dists, 2), axis=1)  # (n_samples, )
            # print("测试数据knn距离：", knn_dists.tolist())
            return knn_dists, knn_neighbors
    def cal_knn_vc(self, samples, sig):  # 计算每个变量的knn贡献度
        knn_dists, knn_neighbors = self.cal_knn_dist(samples, sig)
        knn_vcs = list()  # (n_samples, n_features)
        for i in range(len(knn_dists)):
            neighIds = knn_neighbors[i]
            sample = np.repeat(samples[i].reshape(1,-1), self.k, axis=0)
            vc = np.mean(np.power(sample-self.trainData[neighIds], 2), axis=0)
            knn_vcs.append(vc)
        return knn_dists, np.array(knn_vcs)
    def cal_dist_limit(self, knn_dists):  # 计算knn距离控制限
        dist_limit = np.percentile(knn_dists, self.percent)
        return dist_limit
    def cal_vc_limit(self, knn_vcs):  # 计算变量贡献控制限
        vc_limit = np.percentile(knn_vcs, self.percent, axis=0)  # (n_features, )
        return vc_limit
    def run_vcknn(self):
        self.train_knn_estimator()
        train_dists, train_vcs = self.cal_knn_vc(self.trainData, sig="train")
        self.dist_limit = self.cal_dist_limit(train_dists)
        self.vc_limit = self.cal_vc_limit(train_vcs)
    def test_vcknn(self):
        test_dists, test_vcs = self.cal_knn_vc(self.testData, sig="test")
        self.test_results = list()
        for id in range(len(test_dists)):
            if test_dists[id] <= self.dist_limit:
                self.test_results.append(["normal", id])
            else:
                anomlyId = np.where(test_vcs[id] > self.vc_limit)[0].tolist()
                self.test_results.append(["anomaly", id, anomlyId])
        return self.test_results