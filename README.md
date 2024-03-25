使用 deep Q network 实现金融自动交易，考虑到数据可得性，以虚拟货币为对象。

在传统 DQN 的 Experience Replay 的机制上进行修改，考虑到市场的连续性，将随机抽取改为随机抽取一段连续的市场数据。

Reward 根据总资产的变化计算，GAMMA 设置为 0.997， Learning rate 设置为 1e-3.

尝试多次训练模式后，每 200 段K线的收益大约在 -10% 至 30% 之间浮动，虽然存在基础的套利空间，但风控行为尚不完善，仅作为技术实现之用。
