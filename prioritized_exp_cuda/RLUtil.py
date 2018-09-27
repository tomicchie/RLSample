# -*- coding: utf-8 -*-
r'''
強化学習のサンプルプログラム
'''


class RLUtil:
    '''
    定数などを定義するクラス
    '''
    def __init__(self):
        # 使用する課題名
        self.ENV = 'CartPole-v0'
        # 時間割引率
        self.GAMMA = 0.99
        # 1試行のSTEP数
        self.MAX_STEPS = 200
        # 最大試行回数
        self.NUM_EPISODES = 500
        # Brainクラス用定数
        self.BATCH_SIZE = 32
        self.CAPACITY = 10000
        # Device切り替え用定数
        self.DEVICE = "cpu"
        # TDErrorの誤差に加えるバイアス
        self.TD_ERROR_EPSILON = 0.0001

    def getENV(self):
        '''
        ENVの値を取り出す
        '''
        return self.ENV

    def getGAMMA(self):
        '''
        GAMMAの値を取り出す
        '''
        return self.GAMMA

    def getMAX_STEPS(self):
        '''
        MAX_STEPSの値を取り出す
        '''
        return self.MAX_STEPS

    def getNUM_EPISODES(self):
        '''
        NUM_EPISODESの値を取り出す
        '''
        return self.NUM_EPISODES

    def getBATCH_SIZE(self):
        '''
        BATCH_SIZEの値を取り出す
        '''
        return self.BATCH_SIZE

    def getCAPACITY(self):
        '''
        CAPACITYの値を取り出す
        '''
        return self.CAPACITY

    def getDEVICE(self):
        '''
        DEVICEの値を取り出す
        '''
        return self.DEVICE

    def getTD_ERROR_EPSILON(self):
        '''
        TD_ERROR_EPSILONの値を取り出す
        '''
        return self.TD_ERROR_EPSILON
