# This file calcuate ISC triggering possiblity - displacement curve 

import math
import csv
import numpy as np
from numpy import *
from matplotlib import pyplot as plt

Yfile=["C","I","B","C_test","I_test","B_test"] # Files need to be processed 

def Smin_max(a):
    if len(a) == 1:
        return (a[0], a[0])

    elif len(a) == 2:
        return (min(a), max(a))

    m = len(a) // 2
    lmin, lmax = Smin_max(a[:m])
    rmin, rmax = Smin_max(a[m:])
    return min(lmin, rmin), max(lmax, rmax)

for filename in Yfile:
    curvefile = open(filename+"_curve.csv", "r") # This file save simulation force - displacement curve 
    peakfile = open(filename+"_peak.csv", "r") # This file save ISC triggering forces from repeat tests
    pfile = open(filename+"_Y.csv", "w",encoding='utf-8',newline='')

# stdf = 0.02464

    reader_peakfile = csv.reader(peakfile)
    reader_curvefile = csv.reader(curvefile)

# x1 = np.arange(5).reshape((5, 1))
    peak = []
    curve = []

    for item in reader_peakfile:
        peak.append(float(item[0]))
        # print(item[0])

    for item in reader_curvefile:
        curve.append([float(item[0]),float(item[1])])
        # print(item[0],item[1])

    array_peak = np.array(peak)
    array_curve = np.array(curve)

    displacement = np.zeros(len(array_peak),dtype=float)

    for i in range(len(array_peak)):
        for j in range(len(array_curve)-1):
            if (array_curve[j][1]<array_peak[i]) & (array_curve[j+1][1]>=array_peak[i]):
                displacement[i] = (array_peak[i] - array_curve[j][1]) * (array_curve[j + 1][0] - array_curve[j][0]) / (
                            array_curve[j + 1][1] - array_curve[j][1]) + array_curve[j][0]
            if (array_curve[len(array_curve)-1][1]<array_peak[i]):
                lm=len(array_curve)-1
                displacement[i] = (array_peak[i] - array_curve[lm-1][1]) * (array_curve[lm][0] - array_curve[lm-1][0]) / (
                            array_curve[lm][1] - array_curve[lm-1][1]) + array_curve[lm-1][0]

# print(displacement[i])

    d_mean = mean(displacement)
# print(d_mean)
    d_std = np.std(displacement,ddof=1)
# d_std = stdf * d_mean
# print(d_std/d_mean)

    pd = np.zeros(len(array_curve),dtype=float)
    p = np.zeros(len(array_curve),dtype=float)

    step = 0.05
    p0 = 0

    writer = csv.writer(pfile)

    for i in range(len(array_curve)):
        pd[i] = math.exp(-math.pow(array_curve[i][0]-d_mean,2)/(2*d_std*d_std)) /(d_std * math.pow(2*3.14159,0.5))
        if i==0:
            p[i] = p0
        else:
            p0 = p0 + (pd[i-1]+pd[i])/2 * step
            p[i]= p0
            if p[i] > 0.999999: p[i]=1
            if p[i] <=0.000001: p[i] = 0
        writer.writerow([p[i]])
    # print(i, p[i])

    curvefile.close()
    peakfile.close()
    pfile.close()


# x = np.linspace(0,len(array_curve),len(array_curve))
# y = p
#
# fig=plt.figure(figsize=(150,6))
# plt.plot(x, y,label='First Curve')
# plt.show()










