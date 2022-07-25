# Save the file in the same folder as the sample data files and run

import math
import csv
import numpy as np
from numpy import *
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import time
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

SOC = [0.0, 0.0, 0.0] # SOC of the training cases
filename = ["C_SOC00_E","I_SOC00_E","B_SOC00_E"] # training samples file list

SOC_test = [0.0, 0.0, 0.0] # SOC of the testing cases
filename_test=["C_SOC00_E_test","I_SOC00_E_test","B_SOC00_E_test"] # testing samples file list
testsample = np.array([141,181,181,141,181,181])

# open files
p = []
N = np.zeros([len(filename)])

i = 0
for name in filename:
    pfile = open(name+"_Y.csv", "r")
    reader_pfile = csv.reader(pfile)
    for item in reader_pfile:
        y_temp = float(item[0])
        if y_temp < 0.001 : y_temp = 0.001
        if y_temp > 0.999 : y_temp = 0.999
        Y_linear = -math.log(1/y_temp-1)
        p.append(Y_linear)
        N[i] = N[i] + 1
    i = i + 1
    pfile.close()
Y = np.array(p)

print("training sample", int(sum(N)))

i = 0
d = 11
X = np.zeros([int(sum(N)), d * d * d * d + 1])
n = 0

for name in filename:
    j = 0
    while (j < N[i]):
        sfile = open(name + "_" + str(j + 1) + ".csv", "r")
        reader_sfile = csv.reader(sfile)
        for line in reader_sfile:
            x1 = int(float(line[0]) * 10)
            x2 = int(float(line[1]) * 10)
            x3 = int(float(line[2]) * 10)
            x4 = int(float(line[3]) * 10)
            position = x1 + d * x2 + d * d * x3 + d * d * d * x4
            if (position < d * d * d * d):
                X[n][position + 1] = 1
        sfile.close()
        X[n][0] = SOC[i]
        j = j + 1
        n = n + 1
    i = i + 1

svr = SVR(kernel='rbf', C=500, gamma=0.00001, epsilon=0.01)
# svr = SVR(kernel='sigmoid', gamma=0.001, coef0=0.001)
t0 = time.time()
regr = svr.fit(X, Y)
print(time.time() - t0, "seconds process time")

# cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# scores = cross_val_score(svr, X, Y, cv=cv)
# print(scores)
# print(scores.mean(),scores.std())

i = 0
for name in filename_test:

    Test_sample = testsample[i]
    testfile = name
    print(name)
    Xp = np.zeros([Test_sample, d * d * d * d + 1])
    j = 0
    while (j < Test_sample):
        sfile = open(testfile + "_" + str(j + 1) + ".csv", "r")
        reader_sfile = csv.reader(sfile)
        for line in reader_sfile:
            x1 = int(float(line[0]) * 10)
            x2 = int(float(line[1]) * 10)
            x3 = int(float(line[2]) * 10)
            x4 = int(float(line[3]) * 10)
            position = x1 + d * x2 + d * d * x3 + d * d * d * x4
            if (position < d * d * d * d):
                Xp[j][position + 1] = 1
        sfile.close()
        Xp[j][0] = SOC_test[i]
        j = j + 1

    pt = []
    ofile = open(name + "_Y.csv", "r")
    reader_ofile = csv.reader(ofile)
    for item in reader_ofile:
        pt.append(float(item[0]))
    ofile.close()

    x1 = np.linspace(0, len(Xp)-1, len(Xp))
    y1 = svr.fit(X, Y).predict(Xp)

    for k in range(len(Xp)):
        y1[k] = 1/(1+math.exp(-y1[k]))
        # if y1[k]>1: y1[k] = 1
        # if y1[k]<0: y1[k] = 0

    predictfile = open(filename_test[i]+".csv", "w", encoding='utf-8', newline='')
    writer = csv.writer(predictfile)
    for k in range(len(Xp)-1):
        # if y1[k]>1: y1[k] = 1
        # if y1[k]<0: y1[k] = 0
        x1t = str(x1[k]*0.0005)
        x2t = str(y1[k])
        # x3t = str(pt[k])
        writer.writerow([x1t, x2t])
    predictfile.close()


    x1f = np.linspace(0, (len(Xp) - 1) * 0.05, len(Xp))
    y1f = y1

    x2 = np.linspace(0, (len(pt)-1) * 0.05, len(pt))
    y2 = np.array(pt)
    plt.plot(x1f, y1f, label=name)
    plt.scatter(x2, y2, label=name)

    i = i + 1

Y = np.array(p)

# plt.legend(handles = [l1, l2,], labels = ['a', 'b'], loc = 'best')
plt.show()

# print(r2_score(y1, y2))











