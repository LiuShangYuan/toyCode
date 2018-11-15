import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
每个turn结束后，可以计算该个turn内每个step的reward，有了该reward，可以使用后 -logπ(a|s;w)r_t作为loss计算梯度

'''

class PGN(nn.Module):
    def __init__(self):
        super(PGN, self).__init__()
        self.linear1 = nn.Linear(4, 24)
        self.linear2 = nn.Linear(24, 36)
        self.linear3 = nn.Linear(36, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


class CartAgent(object):
    def __init__(self, learning_rate, gamma):
        self.pgn = PGN()
        self.gamma = gamma

        self._init_memory() ##清空存储单元
        self.optimizer = torch.optim.RMSprop(self.pgn.parameters(), lr=learning_rate)

    def memorize(self, state, action, reward):
        # save to memory for mini-batch gradient descent
        self.state_pool.append(state)
        self.action_pool.append(action)
        self.reward_pool.append(reward)
        self.steps += 1

    def learn(self):
        self._adjust_reward() ### 计算获取当前turn的reward

        # policy gradient
        self.optimizer.zero_grad()
        for i in range(self.steps):
            # all steps in multi games 
            state = self.state_pool[i]
            action = torch.FloatTensor([self.action_pool[i]])
            reward = self.reward_pool[i]

            probs = self.act(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward
            loss.backward()
        self.optimizer.step()
        
        self._init_memory() ### 训练完一轮后清空memory

    def act(self, state):
        return self.pgn(state) 

    def _init_memory(self):
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.steps = 0

    def _adjust_reward(self):
        # backward weight
        ### 通过记录的一个轮次的奖励中，反向计算每个action对应的total reward
        running_add = 0
        for i in reversed(range(self.steps)):
            if self.reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + self.reward_pool[i]
                self.reward_pool[i] = running_add

        # normalize reward
        ### 对reward进行归一化处理
        reward_mean = np.mean(self.reward_pool)
        reward_std = np.std(self.reward_pool)
        for i in range(self.steps):
            self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std


def train():
    # hyper parameter
    BATCH_SIZE = 5
    LEARNING_RATE = 0.01
    GAMMA = 0.99
    NUM_EPISODES = 500

    env = gym.make('CartPole-v1')
    cart_agent = CartAgent(learning_rate=LEARNING_RATE, gamma=GAMMA)

    for i_episode in range(NUM_EPISODES):
        next_state = env.reset()
        env.render(mode='rgb_array')

        for t in count(): ### 一直进行，知道遇到每个终止状态
            state = torch.from_numpy(next_state).float()

            probs = cart_agent.act(state)
            m = Bernoulli(probs)
            action = m.sample()  ### 按照分布采样一个action来执行(自带了探索了利用的策略)

            action = action.data.numpy().astype(int).item()
            next_state, reward, done, _ = env.step(action)
            env.render(mode='rgb_array')

            # end action's reward equals 0
            if done:
                reward = 0

            cart_agent.memorize(state, action, reward) ### 存储当前step的状态，动作的奖励

            if done:
                logger.info({'Episode {}: durations {}'.format(i_episode, t)})
                break

        # update parameter every batch size
        ### 一个turn结束后，当数据够一个batch大小的时候进行参数的update
        if i_episode > 0 and i_episode % BATCH_SIZE == 0:
            cart_agent.learn()


if __name__ == '__main__':
    train()