import numpy as np
import scipy.io as sio
import time
import datetime


class ILFS:
    def __init__(self, X_trn, Y_trn, K, m, delte = 1e-5):
        # a training set X_trn and its label Y_trn
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        # the number of k-nearest neighbor samples used to construct scatter matrixes S and D
        self.K = K
        # the number of features to select
        self.m = m
        # an intended constant that avoids a zero denominator
        self.delte = delte

    # Construct scatter matrixes S and D
    def get_SD(self):
        # Get the size of X_trn
        # n is the number of samples in X_trn
        # d is the number of features that the samples have
        n, d = self.X_trn.shape

        # Apply n * n space to store the distance between samples
        dist = np.zeros((n, n))

        # calculate the distance between each two samples
        for i in range(n):
            # Set the distance of itself to infinity to avoid misuse in subsequent use of k-nearest neighbor
            dist[i][i] = np.inf
            for j in range(i, n):
                dist[i][j] = np.sum((self.X_trn[i] - self.X_trn[j]) ** 2)
                dist[j][i] = dist[i][j]

        # Apply n * n space to store scatter matrixes S,
        # where the K-nearest neighbor with the same label is set to 1
        S = np.zeros((n, n), dtype='int8')
        # Apply n * n space to store scatter matrixes D,
        # where the K-nearest neighbor with the different lable is set to 1
        D = np.zeros((n, n), dtype='int8')

        # Construct scatter matrixes S and D
        for i in range(n):
            for t in range(1, self.K + 1):
                # Find the sample point with the smallest distance from the i-th sample point
                near = np.argmin(dist[i])
                # If these two sample points have the same label, the corresponding position of S is set to 1
                if self.Y_trn[near] == self.Y_trn[i]:
                    S[i][near] = 1
                    S[near][i] = 1
                # If these two sample points have different labels, the corresponding position of D is set to 1
                else:
                    D[i][near] = 1
                    D[near][i] = 1
                # Set the found point distance to infinity
                dist[i][near] = np.inf
        return S, D

    # Do the feature selection
    def select(self):
        # Call function get_SD() to get matrixes S and D
        S, D = self.get_SD()

        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('Matrixes S and D have been constructed:', nowTime)

        # Get the size of X_trn
        # n is the number of samples in X_trn
        # d is the number of features that the samples have
        n, d = self.X_trn.shape

        # The space with length d is applied to store the local distance within the class of each feature
        within = np.zeros(d)
        # The space with length d is applied to store the local distance between the classes of each feature
        between = np.zeros(d)

        # Calculate these two kinds of distances
        for i in range(n):
            for r in range(d):
                # Within class distance only calculates the distance
                # between K-nearest neighbors with the same label, which is set to 1 in S
                within[r] += np.sum((self.X_trn[:, r] - self.X_trn[i][r]) ** 2 * S[i])
                # Within class distance only calculates the distance
                # between K-nearest neighbors with the different labels, which is set to 1 in D
                between[r] += np.sum((self.X_trn[:, r] - self.X_trn[i][r]) ** 2 * D[i])

        # The distance between classes is 0,
        # which indicates that this feature has no discrimination ability
        # and should be excluded in advance to avoid selection
        for r in range(d):
            if between[r] == 0:
                within[r] = np.inf

        # distance between classes of currently selected feature subset
        between_all = 0
        # distance within the class of currently selected feature subset
        within_all = 0
        # indicates vector p. p = 1 means the feature has been selected
        p = np.zeros(d)
        # the order in which features are selected
        order = []

        # Iteratively select until enough features are selected
        for t in range(self.m):
            # The space with length d is applied to store the scores of features in current iteration
            score = np.zeros(d)

            for f in range(d):
                # When p! =0, that is, when the feature is still able to be selected,
                # the score of the current round will be calculated
                if p[f] != 1:
                    score[f] = (between_all + between[f]) / (within_all + within[f] + self.delte)

            # Find the feature with the highest score in this iteration
            max_index = np.argmax(score)
            # If the feature is not selected, it is added to the feature subset
            if p[max_index] == 1:
                break
            p[max_index] = 1
            order.append(max_index)

            # Update the distance between classes of currently selected feature subset
            between_all += between[max_index]
            # Update the distance within the class of currently selected feature subset
            within_all += within[max_index]

        return order


# Read the training set and call ILFS for feature selection
def run(file, K, m=np.inf):
    # Read training set X_trn and its label Y_trn
    data = sio.loadmat(file)
    X_trn = data['X']
    Y_trn = data['Y']

    # Ensure that the number of features to select is less than or equal to the total number of features
    m = min(m, X_trn.shape[1])

    # Output the start time of feature selection
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Experiment begins:', nowTime)
    time_begin = time.time()

    # Initialize a ILFS
    Func = ILFS(X_trn, Y_trn, K, m)
    # Using ILFS to do the feature selection and get the feature order/subset
    order = Func.select()

    # Output the start time of feature selection
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Experiment ends:', nowTime)
    time_end = time.time()

    print('The feature subset or order got by ILFS:', order)
    print('Total time (seconds) of feature selection:', time_end - time_begin)


if __name__ == '__main__':
    # the path of training set
    file = 'Data\\wine.mat'
    # the number of k-nearest neighbor samples used to construct scatter matrixes S and D
    K = 10
    # the number of features to select
    m = 50
    run(file, K, m)
