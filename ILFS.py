import numpy as np
import scipy.io as sio
import os


class ILFS:
    def __init__(self, X_trn, Y_trn, K, m, delte = 1e-5):
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.K = K
        self.m = m
        self.delte = delte

    def get_SD(self):
        n, d = self.X_trn.shape
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                dist[i][j] = np.sum((self.X_trn[i] - self.X_trn[j]) ** 2)
                dist[j][i] = dist[i][j]
        S = np.zeros((n, n))
        D = np.zeros((n, n))
        for i in range(n):
            for t in range(1, self.K + 1):
                near = np.argmin(dist[i])
                if self.Y_trn[near] == self.Y_trn[i]:
                    S[i][near] = 1
                    S[near][i] = 1
                else:
                    D[i][near] = 1
                    D[near][i] = 1
                dist[i][near] = np.inf
        return S, D

    def select(self):
        S, D = self.get_SD()
        n, d = self.X_trn.shape
        within = np.zeros(d)
        between = np.zeros(d)
        for i in range(n):
            for r in range(d):
                within[r] += np.sum((self.X_trn[:, r] - self.X_trn[i][r]) ** 2 * S[i])
                between[r] += np.sum((self.X_trn[:, r] - self.X_trn[i][r]) ** 2 * D[i])

        # The distance between classes is 0,
        # which indicates that this feature has no discrimination ability
        # and should be excluded in advance to avoid selection
        for r in range(d):
            if between[r] == 0:
                within[r] = np.inf

        between_all = 0
        within_all = 0
        p = np.zeros(d)
        order = []
        for t in range(self.m):
            score = np.zeros(d)
            for f in range(d):
                if p[f] != 1:
                    score[f] = (between_all + between[f]) / (within_all + within[f] + self.delte)
            max_index = np.argmax(score)
            if p[max_index] == 1:
                break
            p[max_index] = 1
            order.append(max_index)
            between_all += between[max_index]
            within_all += within[max_index]
        return order


def run(data_path, file, K, m=np.inf):
    data = sio.loadmat(data_path + file)
    X_trn = data['X']
    Y_trn = data['Y']
    m = min(m, X_trn.shape[1])
    Func = ILFS(X_trn, Y_trn, K, m)
    order = Func.select()
    print(order)


if __name__ == '__main__':
    data_path = 'Data\\'
    files = os.listdir(data_path)
    for file in files:
        run(data_path, file, 10, 50)

