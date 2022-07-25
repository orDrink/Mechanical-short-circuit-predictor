# This file obtrain strain information from RADIOSS ouput file .sta file

import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

input_name = "WindingStructureModelV10_offset=0_" # file name
input_number = 141 # total step number
output_name = "C_SOC00_E_"
output_log = "log_C_SOC00_E.csv"

def Smin_max(a):
    if len(a) == 1:
        return (a[0], a[0])

    elif len(a) == 2:
        return (min(a), max(a))

    m = len(a) // 2
    lmin, lmax = Smin_max(a[:m])
    rmin, rmax = Smin_max(a[m:])
    return min(lmin, rmin), max(lmax, rmax)

# open files
log = open(output_log, "w", encoding='utf-8', newline='')

j = 1
N = input_number

# define max min value for each component, run one time this file and check the output, and then modify

x1fmin = -0.88443056981083
x1fmax = 0.13270430380669
x2fmin = -0.65473132930043
x2fmax = 0.094083315604039
x3fmin = -0.076203292505854
x3fmax = 0.36942678816537
x4fmin = -1.0226521061682
x4fmax = 0.95382793019934
x5fmin = -0.72373681413418
x5fmax = 0.60939234074839
x6fmin = -0.60794488717466
x6fmax = 0.74501262011484

ostafile = input_name

while (j<=N):

    print("file", str(j))

    b = len(str(j))

    filename = ''

    if (b == 1): filename = ostafile + "000" + str(int(j)) + ".sta"
    if (b == 2): filename = ostafile + "00" + str(int(j)) + ".sta"
    if (b == 3): filename = ostafile + "0" + str(int(j)) + ".sta"
    if (b == 4): filename = ostafile + str(int(j)) + ".sta"

    stafile = open(filename)

    stressfile = open(output_name + str(j) + ".csv", "w", encoding='utf-8', newline='')

    # initialization
    option = "none"
    straglo = 0
    bstra = []
    sxx = []
    syy = []
    szz = []
    sxy = []
    syz = []
    szx = []
    sp = []
    st1 = []
    st2 = []
    spt = []
    X = []
    xx = 0
    yy = 0
    zz = 0
    xy = 0
    yz = 0
    zx = 0
    for line in stafile:
        #    print(line)
        if (line[0] != "#"):
            if (line[0:14] == "/INIBRI/STRA_F"):
                option = "straglo"
                # print(option)
            # elif (line[0:14] == "/INIBRI/STRA_F"):
            #     option = "none"
            elif (option == "straglo"):
                if (straglo == 0):
                    nn = int(line[0:10])
                    bstra.append(nn)
                    straglo = 1
                elif (straglo == 1):
                    xx = float(line[0:20])
                    yy = float(line[20:40])
                    zz = float(line[40:60])
                    straglo = 2
                elif (straglo == 2):
                    xy = float(line[0:20])
                    yz = float(line[20:40])
                    zx = float(line[40:60])

                    s1t = round((xx - x1fmin) / (x1fmax - x1fmin), 1)
                    s2t = round((yy - x2fmin) / (x2fmax - x2fmin), 1)
                    s3t = round((zz - x3fmin) / (x3fmax - x3fmin), 1)
                    s4t = round((xy - x4fmin) / (x4fmax - x4fmin), 1)
                    s5t = round((yz - x5fmin) / (x3fmax - x5fmin), 1)
                    s6t = round((zx - x6fmin) / (x4fmax - x6fmin), 1)

                    X.append([s1t,s2t,s3t,s4t,s5t,s6t])

                    straglo = 0

    print(len(bstra), "point read")

    # step = 1  # MPa
    # for i in range(len(sp)):
    #     X.append([round(sp[i] / step) * step, round(st1[i] / step) * step, round(st2[i] / step) * step,
    #               round(spt[i] / step) * step])

    X1 = []
    for i in range(len(X)):
        if X[i] not in X1:
            X1.append(X[i])
    
    print(len(X1),"point left")
    
#     enable the following code to check the max, min value

#     X1min, X1max = Smin_max([x[0] for x in X1])
#     X2min, X2max = Smin_max([x[1] for x in X1])
#     X3min, X3max = Smin_max([x[2] for x in X1])
#     X4min, X4max = Smin_max([x[3] for x in X1])
#     X5min, X5max = Smin_max([x[4] for x in X1])
#     X6min, X6max = Smin_max([x[5] for x in X1])

#     print(X1min,X1max)
#     print(X2min,X2max)
#     print(X3min,X3max)
#     print(X4min,X4max)

    writer = csv.writer(stressfile)
    for i in range(len(X1)):
        x1t = str(X1[i][0])
        x2t = str(X1[i][1])
        x3t = str(X1[i][2])
        x4t = str(X1[i][3])
        x5t = str(X1[i][4])
        x6t = str(X1[i][5])
        writer.writerow([x1t, x2t, x3t, x4t, x5t, x6t])

    writer_log = csv.writer(log)
    writer_log.writerow([j,len(X1)])

    # close files
    stafile.close()
    stressfile.close()

    r = 0
    if (int(j/N*100)> r):
        r = int(j/N*100)
        print(int(r), "%")
    print(" ")

    j = j + 1

log.close()
print("finish")




