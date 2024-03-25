import random
import numpy as np
import pandas as pd
import torch.nn.functional
from DQN import EnvOb,Agent

EPOCH = 150000
STEP = 200
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 2000000000
UPDATE_FRENQUENCY = 20

MEMO_SIZE = 10000 # Experience replay的记忆库大小，记忆库在 agent.memo 中实现
MEMOSAMPLE_NUMBER = 64 # 每次从记忆库中抽样的多少

feature_in = 14
action_number = 3
OBSERVATION_SIZE = 200
Initial_amount = 1000

REWARD_BUFFER = np.empty(shape=EPOCH)

file = pd.read_csv("BTCUSDT-1d-2017-2024.csv")
file1 = file.iloc[:5000,:]
state_col = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume','MeanPrice', 'K','D','J']
agent = Agent(input_n=feature_in,output_n=action_number,memory_size=MEMO_SIZE)
env = EnvOb(file=file1,state_col=state_col,observation_size=STEP-1)

Sc = 1
ranflag = []
q_values_li = []
for epoch_i in range(EPOCH):
    s = env.reset()
    reward = 0
    reward = torch.tensor(reward,dtype=torch.float32)
    for step_i in range(STEP):
        EPSILON = np.interp(epoch_i*STEP+step_i,[0,EPSILON_DECAY],[EPSILON_START,EPSILON_END])

        if random.random() <= EPSILON:
            action = env.action_sample()
        else:
            try:
                action = agent.qnet.act(s)
                ranflag.append(epoch_i)
            except RuntimeError:
                print(s,type(s))
                input()

        s,r,s_,done = env.step(action)
        step_r = r - reward
        reward = r
        s = s_
        agent.memo.addMemo((s,step_r,action,s_,done))

        if done:
            REWARD_BUFFER[epoch_i] = reward
            #print("After done, Reward: ",reward)
            break

        if epoch_i > 145000:
            PATH = './test_state_dict' + str(Sc) + '.pth'
            torch.save(agent.qnet.state_dict(), PATH)
            s = env.reset(test=True)
            reward = 0
            file = file
            count = 0
            reward_buffer = []
            while True:
                a = agent.qnet.act(s)
                print('Action: ',a)
                s, r, s_, done = env.step(a)
                reward += r
                count += 1
                if done == 1:
                    s = env.reset(test=True)
                    reward = 0
                    reward_buffer.append(reward)
                    print("Avg reward: ",np.mean(reward_buffer))


        #print('s: ',step_i,end=' ')
        # 在开始时，terminate point 设置正确，但在抽样时，terminate point 并不一定与一开始一致
        # 在抽样时，应当随机抽取连续的 n 时间走势，独立计算是否抵达 terminate point
        # batch_a 中的每一项元素都应当为连续的动作序列，其余几项也应为连续序列
        batch_s,batch_r,batch_a,batch_s_,batch_done = agent.memo.sample(MEMOSAMPLE_NUMBER)

        target_q_values = agent.target_net(batch_s_)
        max_target_q = target_q_values.max(dim=1,keepdim=True)[0]
        Target_Value = reward + agent.GAMMA * (1-done) * max_target_q

        q_values = agent.qnet(batch_s)
        q_values_li.append(q_values.max(dim=1,keepdim=True).values[0].item())
        try:
            max_q_value = q_values.gather(dim=1,index=batch_a)
            #max_q_value = q_values.max(dim=1,keepdim=True)[0]
        except:
            #print(batch_a)
            input('error')

        loss = torch.nn.functional.smooth_l1_loss(Target_Value,max_q_value)

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    if np.mean(REWARD_BUFFER[epoch_i-10:epoch_i]) > 200:
        if Sc == 1:
            PATH = './test_state_dict'+str(Sc)+'.pth'
            torch.save(agent.qnet.state_dict(), PATH)
        print("Success ", Sc, " Reward ", reward)
        Sc += 1

    if epoch_i % UPDATE_FRENQUENCY == 0:
        agent.target_net.load_state_dict(agent.qnet.state_dict())

        print("EPOCH ",epoch_i,'/',EPOCH)
        print("10 AVG reward: ",np.mean(REWARD_BUFFER[epoch_i-10:epoch_i]))
        print("10 time rewards:", end=' ')
        for i in range(epoch_i-10,epoch_i):
            if i in ranflag:
                print("%.3f"%(REWARD_BUFFER[i]),end='* ')
            else:
                print("%.3f"%REWARD_BUFFER[i], end=' ')
        print()
        print("20 times avg q values:",np.mean(q_values_li[-20:]))
        print("===================")
        print()
        ranflag = []

