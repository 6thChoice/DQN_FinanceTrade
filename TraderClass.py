import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Trader:
    def __init__(self,amount,file):
        self.InitialAmount = amount
        self.amount = amount
        self.numberOfCurrency = 0
        self.recordForAsset = [amount for i in range(len(file))]
        self.recordForDate = []
        self.file = file
        self.price = file['MeanPrice']
        #self.date = file['open_time']
        self.buyPrice = -1
        self.down = False

    def totalAsset(self,index):
        self.recordForAsset[index] = self.amount+self.numberOfCurrency*self.price[index]
        return self.amount+self.numberOfCurrency*self.price[index]

    def buy(self,index,quota):
        # quota 为买入额占持有货币的比重
        if quota <= 1:
            self.numberOfCurrency = quota*self.amount / self.price[index]
            self.amount -= quota*self.amount
            self.recordForDate.append(index)
            self.buyPrice = self.price[index]
            #print("Buy in day ", self.date[index])
            #print("The price ", self.price[index])
            #print("Current Asset: ", self.totalAsset(index))
            #print("The KDJ is (", self.file['K'][index], self.file['D'][index], self.file['J'][index], ')')
            #print('----------------------------------------')
            #print()

    def sell(self,index,quotacur):
        # quotacur 为卖出额占持有虚拟货币的比重
        if self.numberOfCurrency != 0 and quotacur <= 1:
            self.amount += self.numberOfCurrency * self.price[index]
            self.numberOfCurrency = 0
            for i in range(index,len(self.file)):
                self.recordForAsset[i] = self.totalAsset(index)
            self.recordForDate.append(index)
            self.buyPrice = -1

            #print("Sell in day ", self.date[index])
            #print("The price ", self.price[index])
            #print("The quota: ",quotacur)
            #print("Current Asset: ", self.totalAsset(index))
            #print("The KDJ is (",self.file['K'][index],self.file['D'][index],self.file['J'][index],')')
            #print("=========================================")
            #print()

    def show(self):
        time = self.date
        print("Initial Amount: ",self.InitialAmount)
        print("Actual Amount: ",self.amount)
        print("Total Asset: ",self.totalAsset(len(self.file)-1))
        print("Income rate: ",(self.totalAsset(len(self.file)-1)-self.InitialAmount)/self.InitialAmount * 100,'%')
        fig,ax = plt.subplots(1,1)
        fig.suptitle('Total Asset Change Chart')
        ax.plot(time,self.recordForAsset)
        ticks = []
        for i in range(len(time)):
            if i in self.recordForDate:
                ticks.append(time[i].split('-')[1]+'-'+time[i].split('-')[2])
            else:
                ticks.append(' ')
        plt.xticks(time,ticks)
        plt.show()

class KDJclassifier:
    def __init__(self,file,quantile):
        # quantile 为分位数比例，如0.2，0.3等，此处不应超过 0.5
        self.file = file
        self.stageBuy = False
        self.stageSell = False
        self.k = self.file['K']
        self.d = self.file['D']
        self.j = self.file['J']
        self.kQlow = self.k.quantile(quantile)
        self.dQlow = self.d.quantile(quantile)
        self.kQhigh = self.k.quantile(1-quantile)
        self.dQhigh = self.d.quantile(1-quantile)
        self.stackBuy = []
        self.stackSell = []
        #print("KQlow: ",self.kQlow)
        #print("KQhigh: ",self.kQhigh)
        #print("DQlow: ",self.dQlow)
        #print("DQhigh: ",self.dQhigh)
        #print()

    def detect(self,index):
        if self.k[index] < self.kQlow and self.d[index] < self.dQlow:
            if self.stackBuy == []:
                self.stageBuy = True
                self.stackBuy.append(1)
        else:
            self.stageBuy = False
            self.stackBuy = []
        if self.k[index] > self.kQhigh and self.d[index] > self.dQhigh:
            if self.stackSell == []:
                self.stageSell = True
                self.stackSell.append(1)
        else:
            self.stageSell = False
            self.stackSell = []
        return self.stageBuy, self.stageSell

    def show(self):
        print("Klow: ",self.kQlow)
        print("Khigh: ",self.kQhigh)
        print("Dlow: ",self.dQlow)
        print("Dhigh: ",self.dQhigh)