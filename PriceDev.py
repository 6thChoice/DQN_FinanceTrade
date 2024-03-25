import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pandas as pd

# 对已合并的文件进行数据调整

fileName = "BTCUSDT-1d-2017-2024.csv"
file = pd.read_csv(fileName)

# 计算价格变化量

meanPrice = file['MeanPrice']
priceChange = []
for i in range(len(meanPrice)):
    if i + 1 < len(meanPrice):
        priceChange.append(file['MeanPrice'][i+1] - file['MeanPrice'][i])
priceChange.append(0)
file['PriceChange'] = priceChange
#file.to_csv(fileName,index=False)


# 计算 KDJ 指标
def getMin(list):
    min = 10000
    for i in range(len(list)):
        if min > list[i]:
            min = list[i]
    return min
def getMax(list):
    max = 0
    for i in range(len(list)):
        if max < list[i]:
            max = list[i]
    return max

closeList = file['close']
lowList = file['low']
highList = file['high']
time = file['open_time']
closeList = list(closeList)
lowList = list(lowList)
highList = list(highList)
time = list(time)
recentNList = []
maxNList = []
minNList = []
closePrice = 0
minN = 10000
maxN = 0
N = 9
k = []
d = []
j = []
for i in range(len(closeList)):
    count = len(recentNList)
    maxN = getMax(maxNList)
    minN = getMin(minNList)
    if count < N:
        recentNList.append(closeList[i])
        maxNList.append(highList[i])
        minNList.append(lowList[i])
        k.append(50)
        d.append(50)
        j.append(50)
        continue

    closePrice = recentNList[-1]
    RSV = (closePrice-minN) / (maxN-minN) * 100
    K = (2/3) * k[i-1] + (1/3) * RSV
    k.append(K)
    D = (2/3) * d[i-1] + (1/3) * K
    d.append(D)
    J = 3 * K - 2 * D
    j.append(J)

    recentNList.append(closeList[i])
    recentNList.pop(0)
    maxNList.append(highList[i])
    maxNList.pop(0)
    minNList.append(lowList[i])
    minNList.pop(0)
    maxN = getMax(maxNList)
    minN = getMin(minNList)

k.pop(0)
d.pop(0)
j.pop(0)
k.append(np.mean(k))
d.append(np.mean(d))
j.append(np.mean(j))
file['K'] = k
file['D'] = d
file['J'] = j

file.to_csv(fileName,index=False)