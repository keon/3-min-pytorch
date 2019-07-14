#!/usr/bin/env python
# coding: utf-8

# # 카트폴 게임 마스터하기

import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import numpy as np


# ## OpenAI Gym을 이용하여 게임환경 구축하기
# 강화학습 예제들을 보면 항상 게임과 연관되어 있습니다. 원래 우리가 궁극적으로 원하는 목표는 어디서든 적응할 수 있는 인공지능이지만, 너무 복잡한 문제이기도 하고 가상 환경을 설계하기도 어렵기 때문에 일단 게임이라는 환경을 사용해 하는 것입니다.
# 대부분의 게임은 점수 혹은 목표가 있습니다. 점수가 오르거나 목표에 도달하면 일종의 리워드를 받고 원치 않은 행동을 할때는 마이너스 리워드를 주는 경우도 있습니다. 아까 비유를 들었던 달리기를 배울때의 경우를 예로 들면 총 나아간 길이 혹은 목표 도착지 도착 여부로 리워드를 주고 넘어질때 패널티를 줄 수 있을 것입니다. 
# 게임중에서도 가장 간단한 카트폴이라는 환경을 구축하여 강화학습을 배울 토대를 마련해보겠습니다.

env = gym.make('CartPole-v1')


# ### 하이퍼파라미터
# 하이퍼파라미터
EPISODES = 50    # 에피소드 반복 횟수
EPS_START = 0.9  # e-greedy threshold 시작 값
EPS_END = 0.05   # e-greedy threshold 최종 값
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8      LR = 0.001       # NN optimizer learning rate
BATCH_SIZE = 64  # Q-learning batch size


# ## DQN 에이전트

class DQNAgent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.steps_done = 0
    
    def act(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            return self.model(state).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(2)]])

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state])))
    
    def learn(self):
        """Experience Replay"""
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        current_q = self.model(states).gather(1, actions)
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (GAMMA * max_next_q)
        
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ## 학습 준비하기
# 드디어 만들어둔 DQNAgent를 인스턴스화 합니다.
# 그리고 `gym`을 이용하여 `CartPole-v0`환경도 준비합니다.
# 자, 이제 `agent` 객체를 이용하여 `CartPole-v0` 환경과 상호작용을 통해 게임을 배우도록 하겠습니다.
# 학습 진행을 기록하기 위해 `score_history` 리스트를 이용하여 점수를 저장하겠습니다.

agent = DQNAgent()
env = gym.make('CartPole-v0')
score_history = []


# ## 학습 시작

for e in range(1, EPISODES+1):
    state = env.reset()
    steps = 0
    while True:
        env.render()
        state = torch.FloatTensor([state])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action.item())

        # negative reward when attempt ends
        if done:
            reward = -1

        agent.memorize(state, action, reward, next_state)
        agent.learn()

        state = next_state
        steps += 1

        if done:
            print("에피소드:{0} 점수: {1}".format(e, steps))
            score_history.append(steps)
            break




