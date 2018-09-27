# -*- coding: utf-8 -*-
'''
強化学習のサンプルプログラム
'''
import RLBrain

class RLAgent:
    '''
    cartpoleで動くAgentクラス
    '''
    def __init__(self, num_states, num_actions):
        '''
        課題の状態と行動の数を設定する
        '''
        # Brainクラスのインスタンスを生成
        self.brain = RLBrain.RLBrain(num_states, num_actions)

    def update_q_function(self):
        '''
        Q関数を更新する
        '''
        self.brain.replay()

    def get_action(self, state, episode):
        '''
        行動を決定する
        '''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''
        memoryオブジェクトにstate, action, state_next, rewardの内容を保存する
        '''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        '''
        Target Q-NetworkをMain Q-Networkと同じに更新
        '''
        self.brain.update_target_q_network()