import random
from TraderClass import Trader
import numpy as np
import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self,input_n,output_n):
        super(Network, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.model = nn.Sequential(
            nn.Linear(input_n,20),
            #nn.Tanh(),
            nn.LeakyReLU(0.02),
            nn.Linear(20,20),
            nn.LeakyReLU(0.02),
            nn.Linear(20,output_n)
        )

    def forward(self,input):
        input = input.float()
        return self.model(input)

    def act(self,s):
        q_values = self.forward(s)
        q_values = torch.as_tensor(q_values)
        action = torch.argmax(q_values).detach().item()
        return action

class MemoryBuffer:
    def __init__(self,size,n_s,n_a):
        self.Buffer_size = size
        self.all_a = np.asarray([0 for i in range(size)])
        self.all_s = np.empty(shape=(size,n_s),dtype=np.float64)
        self.all_r = np.empty(shape=size,dtype=np.float64)
        self.all_a = np.asarray([0 for i in range(size)])
        #self.all_a = np.empty(shape=size,dtype=np.int64)
        self.all_s_ = np.empty(shape=(size,n_s),dtype=np.float64)
        self.all_done = np.empty(shape=size,dtype=np.int64)
        self.idx = 0
        self.count = 0


    def addMemo(self,data):
        s = data[0]
        r = data[1]
        a = data[2]
        s_ = data[3]
        done = data[4]

        self.all_s[self.idx] = s
        self.all_r[self.idx] = r
        self.all_a[self.idx] = a
        self.all_s_[self.idx] = s_
        self.all_done[self.idx] = done

        self.idx = (self.idx+1) % self.Buffer_size
        self.count += 1


    def sample(self,sample_size):
        if self.count < sample_size:
            idxes = range(0,self.count)
        else:
            idxes = random.sample(range(0,self.Buffer_size),sample_size)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []
        for idx in idxes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s),dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a),dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r),dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_),dtype=torch.float32)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done),dtype=torch.float32).unsqueeze(-1)

        """for item in batch_a:
            if item > 10000:
                print(batch_a)
                for i in self.all_a:
                    print(i,end=',')
                print()
                print(self.idx,self.count)"""
        return batch_s_tensor,batch_r_tensor,batch_a_tensor,batch_s__tensor,batch_done_tensor

class EnvOb:
    def __init__(self,file,state_col,observation_size):
        self.state = file
        for c in file.columns:
            if c not in state_col:
                self.state.drop(columns=c,axis=1,inplace=True)

        self.idx = 0
        self.ob_size = observation_size # 观察范围，决定是否抵达 terminate point
        self.terminate_idx = 0
        self.action = [0,1,2] # 0:buy 1:none 2:sell
        self.full = 0 # 仓位标志

        self.InitialAmount = 1000
        self.trader = Trader(amount=self.InitialAmount,file=self.state)
        self.totalAsset = self.InitialAmount

    def reset(self,test=False):
        if test == True:
            self.idx = 0
            self.ob_size = len(self.state)-50
        else:
            self.idx = 0
            #self.ob_size = random.randint(15,25)
            #self.idx = random.randint(0,len(self.state)-50)
        self.terminate_idx = self.idx + self.ob_size
        self.trader = Trader(amount=1000,file=self.state)
        self.totalAsset = 1000
        return torch.tensor(data=self.state.iloc[self.idx,:].values)

    def action_sample(self): # 随机返回一种可执行动作，但并不直接执行
        if self.full == 0:
            idx = random.randint(0,1)
        if self.full == 1:
            idx = random.randint(1,2)
        action = np.array([self.action[idx]])
        action = torch.tensor(action,dtype=torch.int64)
        return torch.as_tensor(action,dtype=torch.int64)

    def GetReward(self,index):
        flag = True
        if self.totalAsset <= self.InitialAmount / 10:
            flag = False
        if self.full == 0:
            return torch.tensor(0),flag
        if self.full == 1:
            #reward = self.trader.totalAsset(self.idx) - self.totalAsset
            self.totalAsset = self.trader.totalAsset(self.idx)
            reward = (self.totalAsset-self.InitialAmount) / (self.InitialAmount /1000)
            return torch.tensor(reward,dtype=torch.float32),flag

    def step(self,action): # 不设金额上限
        if action == 0: # buy
            self.full = 1
            self.trader.buy(self.idx,quota=1)
        if action == 2: # sell
            self.full = 0
            self.trader.sell(self.idx,quotacur=1)

        state = torch.tensor(self.state.iloc[self.idx,:],dtype=torch.float32)
        flag = True
        reward,flag = self.GetReward(self.idx)
        if self.idx < self.terminate_idx and flag == True:
            next_state = torch.tensor(self.state.iloc[self.idx+1,:],dtype=torch.float32)
            self.idx += 1
            done = 0 # 是否抵达 terminate point
        else:
            next_state = state
            done = 1
            #if self.totalAsset < 1010:
                #reward -= 10


        return state,reward,next_state,done


class Agent:
    def __init__(self,input_n,output_n,memory_size):
        self.memo = MemoryBuffer(size=memory_size,n_s=input_n,n_a=output_n)
        self.qnet = Network(input_n=input_n,output_n=output_n)
        self.target_net = Network(input_n=input_n,output_n=output_n)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.lr  = 1e-3
        self.GAMMA = 0.999

        self.optimizer = torch.optim.Adam(params=self.qnet.parameters(),lr=self.lr)
