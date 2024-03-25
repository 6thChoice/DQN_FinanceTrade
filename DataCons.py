import pandas as pd
import numpy as np
import os
import datetime

# 对尚未合并的各文件进行整合规整

# 为缺少列名的表格添加列名
standard_column = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']

dirPath = 'D:\PyCharm Community Edition 2022.1.2\Binance\BTCUSDT-1d-2017-2024'
dirList = os.listdir(dirPath)
for item in dirList:
    fileName = dirPath + '\\' + item
    file = pd.read_csv(fileName)

    # 添加列名
    column = list(file.columns)
    if column[0].isdigit():
        file = pd.read_csv(fileName, header=None)
        data = {}
        for i in range(len(standard_column)):
            data[standard_column[i]] = file.iloc[:,i]
        data = pd.DataFrame(data)
        #data.to_csv(fileName,index=False)
    else:
        data = file

    # 修改时间格式
    data['open_time'] = data['open_time'] / 1000
    time_list = []
    for i in range(len(file)):
        time_list.append(datetime.datetime.fromtimestamp(data['open_time'][i]))
    data['open_time'] = time_list
    #data.to_csv(fileName,index=False)
    #time_list = []
    #for i in range(len(data)):
    #    time_list.append(str(data['open_time'][i]).split(' ')[0])
    #data['open_time'] = time_list
    #file.to_csv(fileName,index=False)


    # 添加最高价与最低价的平均值
    high_price = data['high']
    low_price = data['low']
    mean_price = (high_price + low_price) / 2
    data['MeanPrice'] = mean_price
    data.to_csv(fileName,index=False)

# 文件整合

dirPath = 'D:\PyCharm Community Edition 2022.1.2\Binance\BTCUSDT-1d-2017-2024'
dirList = os.listdir(dirPath)
s = True
for item in dirList:
    fileName = dirPath + '\\' + item
    file = pd.read_csv(fileName)
    file.to_csv("BTCUSDT-1d-2017-2024.csv", index=False, header=s, mode='a+')
    s = False