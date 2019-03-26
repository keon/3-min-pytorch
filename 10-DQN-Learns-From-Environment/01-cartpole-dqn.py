#!/usr/bin/env python
# coding: utf-8

# # 카트폴 게임 마스터하기
# 어떤 게임을 마스터한다는 뜻은 최고의 점수를 받는다는 뜻이기도 합니다.
# 그러므로 게임의 점수를 리워드로 취급하면 될 것 같습니다.
# 우리가 만들 에이전트는 리워드를 예측하고,
# 리워드를 최대로 만드는 쪽으로 학습하게 할 것입니다.
# 예를들어 카트폴 게임에서는 막대기를 세우고 오래 버틸수록 점수가 증가합니다.
# 카트폴 게임에서 막대가 오른쪽으로 기울었을때,
# 어느 동작이 가장 큰 리워드를 준다고 예측할 수 있을까요?
# 오른쪽으로 가서 중심을 다시 맞춰야 하니
# 오른쪽 버튼을 누르는 쪽이 왼쪽 버튼보다 리워드가 클 것이라고 예측 할 수 있습니다.
# 이것을 한줄로 요약하자면 아래 한줄의 코드가 됩니다.
# ```
# target = reward + gamma * np.amax(model.predict(next_state))
# ```
# DQN은 가장 중요한 특징 2가지로 요약될 수 있습니다.
# 바로 기억하기(Remember)와 다시 보기(Replay)입니다.
# 둘다 간단한 아이디어이지만 신경망이 강화학습에 이용될 수 있게 만든 혁명적인 방법들입니다.
# 순서대로 개념과 구현법을 알아보도록 하겠습니다.

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
# hyper parameters
EPISODES = 50  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
BATCH_SIZE = 64  # Q-learning batch size


# ## DQN 에이전트
# DQNAgent라는 클래스를 만들어 
# ```python
# class DQNAgent
# ```
# ### DQN 에이전트의 뇌, 뉴럴넷
# ![dqn_net](./assets/dqn_net.png)
# ```python
# self.model = nn.Sequential(
#     nn.Linear(4, 256),
#     nn.ReLU(),
#     nn.Linear(256, 2)
# )
# ```
# ### 행동하기 (Act)
# ### 전 경험 기억하기 (Remember)
# 신경망을 Q-learning학습에 처음 적용하면서 맞닥뜨린 문제는
# 바로 신경망이 새로운 경험을 전 경험에 겹쳐쓰며 쉽게 잊어버린다는 것이었습니다.
# 그래서 나온 해결책이 바로 기억하기(Remember)라는 기능인데요,
# 바로 이전 경험들을 배열에 담아 계속 재학습 시키며 신경망이 까먹지 않게 하는 아이디어 입니다. 
# 각 경험은 상태, 행동, 보상등을 담아야 합니다.
# 이전 경험들을 담을 배열을 `memory`라고 부르고 아래와 같이 만들어봅시다.
# ```python
# self.memory = [(상태, 행동, 보상, 다음 상태)...]
# ```
# 이를 구현하기 위해 복잡한 모델을 만들때는 Memory클래스를 구현하기도 하지만,
# 이번 예제에서는 사용하기 가장 간단한 deque (double ended queue),
# 즉 큐(queue) 자료구조를 이용할 것입니다.
# 파이썬에서 `deque`의 `maxlen`을 지정해주었을때 큐가 가득 찼을 경우
# 제일 오래된 요소부터 없어지므로
# 자연스레 오래된 기억을 까먹게 해주는 역할을 할 수 있습니다.
# ```python
# self.memory = deque(maxlen=10000)
# ```
# 그리고 memory 배열에 새로운 경험을 덧붙일 remember() 함수를 만들어보겠습니다.
# ```python
# def memorize(self, state, action, reward, next_state):
#     self.memory.append((state,
#                         action,
#                         torch.FloatTensor([reward]),
#                         torch.FloatTensor([next_state])))
# ```
# ### 경험으로부터 배우기 (Experience Replay)
# 이전 경험들을 모아놨으면 반복적으로 학습해야합니다.
# 사람도 수면중일때 자동차 운전, 농구 슈팅,
# 등 운동과 관련된 정보를 정리하며,
# 단기 기억을 운동피질에서 측두엽으로 전달하여 장기 기억으로 변환시킨다고 합니다.
# 우연하게도 DQN에이전트가 기억하고 다시 상기하는 과정도 비슷한 개념입니다.
# `learn`함수는 바로 이런 개념으로 방금 만들어둔 뉴럴넷인 `model`을
# `memory`에 쌓인 경험을 토대로 학습시키는 역할을 합니다.
# ```python
#    def learn(self):
#         """Experience Replay"""
#         if len(self.memory) < BATCH_SIZE:
#             return
#         batch = random.sample(self.memory, BATCH_SIZE)
#         states, actions, rewards, next_states = zip(*batch)
# ```
# `self.memory`에서 무작위로 배치 크기만큼의 "경험"들을 가져옵니다.
# 이 예제에선 배치사이즈를 64개로 정했습니다.
# ```python
#         states = torch.cat(states)
#         actions = torch.cat(actions)
#         rewards = torch.cat(rewards)
#         next_states = torch.cat(next_states)
# ```
# 각각의 경험들은 상태(`states`), 행동(`actions`), 행동에 따른 보상(`rewards`),
# 그리고 다음 상태(`next_states`)를 담고있습니다.
# 모두 리스트의 리스트 형태이므로 `torch.cat()`을 이용하여 하나의 리스트로 만듭니다.
# `cat`은 concatenate의 준말로 결합하다, 혹은 연결하다라는 뜻입니다.
# ```python
#         current_q = self.model(states).gather(1, actions)
#         max_next_q = self.model(next_states).detach().max(1)[0]
#         expected_q = rewards + (GAMMA * max_next_q)
# ```
# Q값을 구합니다.
# ```python
#         loss = F.mse_loss(current_q.squeeze(), expected_q)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
# ```
# 학습시킵니다.

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


# ## 학습 시작하기
# EPISODES는 얼마나 많은 게임을 진행하느냐를 나타내는 하이퍼파라미터입니다.
# ```
# for e in range(1, EPISODES+1):
#     state = env.reset()
#     steps = 0
# ```
# `done`변수에는 게임이 끝났는지의 여부가 참(True), 거짓(False)로 표현됩니다.
# ```
#     while True:
#         env.render()
#         state = torch.FloatTensor([state])
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action.item())
# ```
# 우리의 에이전트가 한 행동의 결과가 나왔습니다!
# 이 경험을 기억(memorize)하고 배우도록합니다.
# ```
#         # negative reward when attempt ends
#         if done:
#             reward = -1
#         agent.memorize(state, action, reward, next_state)
#         agent.learn()
#         state = next_state
#         steps += 1
# ```
# 게임이 끝났을 경우 `done`이 `True`가 되며 아래 코드가 실행되게 됩니다.
# 보통 게임 분석을 위해 복잡한 도구와 코드가 사용되는 경우가 많으나
# 여기서는 간단하게 에피소드 숫자와 점수만 표기하도록 하겠습니다.
# 또 앞서 만들어둔 `score_history` 리스트에 점수를 담도록 합니다.
# 마지막으로 게임이 더 이상 진행되지 않으므로 `break` 문으로 무한루프를 나옵니다.
# ```
#         if done:
#             print("에피소드:{0} 점수: {1}".format(e, steps))
#             score_history.append(steps)
#             break
# ```

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




