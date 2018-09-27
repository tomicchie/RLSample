# -*- coding: utf-8 -*-
r'''
強化学習のサンプルプログラム
'''
from collections import namedtuple

import random
import RLUtil

class ReplayMemory:
    '''
    経験を保存するメモリクラスを定義
    '''

    def __init__(self, capacity):
        # メモリの最大長さ
        self.capacity = capacity
        # 経験を保存する変数
        self.memory = []
        # 保存するindexを示す変数
        self.index = 0
        # Utilクラスインスタンス生成
        self.util = RLUtil.RLUtil()


    def push(self, state, action, state_next, reward):
        '''
            上記で記述したnamedtupleのTransitionをメモリ上に保存する
        '''
        if len(self.memory) < self.capacity:
            # メモリが満タンでない場合は足す
            self.memory.append(None)

        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存する
        self.memory[self.index] = Transition(state, action, state_next, reward)

        # 保存するindexを1つずらす
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        '''
        batch_size分だけ、ランダムに保存内容を取り出す
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''
        関数lenに対して、現在の変数memoryの長さを返す
        '''
        return len(self.memory)
