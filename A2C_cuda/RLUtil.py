# -*- coding: utf-8 -*-
'''
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
        self.NUM_EPISODES = 1000
        # Brainクラス用定数
        self.BATCH_SIZE = 32
        self.CAPACITY = 10000
        # Device切り替え用定数
        self.DEVICE = "cpu"
        # TDErrorの誤差に加えるバイアス
        self.TD_ERROR_EPSILON = 0.0001

        # A2C用定数
        # 同時に実行する環境
        self.NUM_PROCESSES = 16
        # 何ステップ勧めて報酬和を計算するのかの設定
        self.NUM_ADVANCED_STEP = 5
        # A2C誤差関数計算用の定数
        self.VALUE_LOSS_COEF = 0.5
        self.ENTROPY_COEF = 0.01
        self.MAX_GRAD_NORM = 0.5

    # getter関数群。privateではないけどそれっぽく。

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
    
    def getNUM_PROCESSES(self):
        '''
        NUM_PROCESSESの値を取り出す
        '''
        return self.NUM_PROCESSES
    
    def getNUM_ADVANCED_STEP(self):
        '''
        NUM_ADVANCED_STEPの値を取り出す
        '''
        return self.NUM_ADVANCED_STEP
    
    def getVALUE_LOSS_COEF(self):
        '''
        VALUE_LOSS_COEFの値を取り出す
        '''
        return self.VALUE_LOSS_COEF
    
    def getENTROPY_COEF(self):
        '''
        ENTROPY_COEFの値を取り出す
        '''
        return self.ENTROPY_COEF
    
    def getMAX_GRAD_NORM(self):
        '''
        MAX_GRAD_NORMの値を取り出す
        '''
        return self.MAX_GRAD_NORM
