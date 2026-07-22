import os
import sys
import imageio
from env_poly_SNN import Maze
from env import Maze2
from RL_brain import QLearningTable
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(1)
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
import torch, os, sys
from torch import nn
from torch.nn import Parameter
import abc
import math
from abc import ABC
import torch.nn.functional as F
from braincog.base.node.node import *
from braincog.base.learningrule.STDP import *
from braincog.base.connection.CustomLinear import *
from braincog.base.utils.visualization import spike_rate_vis, spike_rate_vis_1d#, spike_vis_2, spike_vis_5

class BrainArea(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()
      
    @abc.abstractmethod
    def forward(self, x):
        """
        Calculate the forward propagation process
        :return:x is spike
        """
        return x

    def reset(self):
        """
        Calculate the forward propagation process
        :return:x is spike
        """
        pass

class BAESNN(BrainArea):
    """
    Affactive Empathy Network
    """

    def __init__(self,):
        super().__init__()

        self.node = [IFNode() for i in range(5)]
        
        self.connection = []
        
        con_matrix0 = torch.eye(24, 24)*6
        self.connection.append(CustomLinear(con_matrix0))#input-emotion
        
        con_matrix1 = torch.eye(24, 24)
       
        self.connection.append(CustomLinear(con_matrix1))#emotion-ifg
        
        con_matrix2 = torch.zeros((24, 24), dtype=torch.float)   
        self.connection.append(CustomLinear(con_matrix2))#perception-ifg
        
        con_matrix3 = torch.eye(24, 24)*6
        self.connection.append(CustomLinear(con_matrix3))#input-perception
        
        con_matrix4=torch.zeros((24,10), dtype=torch.float)
        for j in range(10):
            if j in np.arange(0,5,1):
                for i in np.arange(0, 12, 1):
                    con_matrix4[i,j] =2
            if j in np.arange(5,10,1):
                for i in np.arange(12, 24, 1):
                    con_matrix4[i,j] =2
        self.connection.append(CustomLinear(con_matrix4))#emotion-sma
        
        con_matrix5=torch.zeros((24,10), dtype=torch.float)
        self.connection.append(CustomLinear(con_matrix5))#perception-m1
        
        con_matrix6 = torch.eye(10, 10)*6
        self.connection.append(CustomLinear(con_matrix6))#sma-m1
        
        self.stdp = []
        self.stdp.append(STDP(self.node[0], self.connection[0]))#0 node0 emotion
        self.stdp.append(STDP(self.node[2], self.connection[3]))#1 node2 perception
        self.stdp.append(MutliInputSTDP(self.node[1], [self.connection[1], self.connection[2]]))#2 node1 ifg
        self.stdp.append(MutliInputSTDP(self.node[3], [self.connection[4], self.connection[5]]))#3 node3 sma
        self.stdp.append(STDP(self.node[4], self.connection[6]))#4 node4 m1
        self.stdp.append(STDP(self.node[1],self.connection[2]))#5 node1 ifg
        self.stdp.append(STDP(self.node[3],self.connection[5]))#6 node3 sma
    def forward(self, x1,x2):
        out__m, dw0 = self.stdp[0](x1)#node0 emotion
        out__p, dw3 = self.stdp[1](x2)#node2 perception
        out__ifg,dw_p_i=self.stdp[2](out__m,out__p)#node1 ifg   
        out__sma,dw_p_s=self.stdp[3](out__m,out__p)#node3  sma
        out__m1,dw1=self.stdp[4](out__sma)#node4 m1
    
        return dw_p_i,dw_p_s,out__ifg,out__sma,out__m1,out__m,out__p
    
    def empathy(self,x3):
        out_p,dw2=self.stdp[1](x3)#node2 perception
        out_ifg,dw4=self.stdp[5](out_p)#node1 ifg
        out_sma,dw5=self.stdp[6](out_p)#node3 sma
        out_m1,dw6=self.stdp[4](out_sma)#node4 m1
        return out_ifg,out_sma,out_m1,out_p
        
    def UpdateWeight(self, i, dw, delta):
        self.connection[i].update(dw*delta)
        self.connection[i].weight.data= torch.clamp(self.connection[i].weight.data,-1,4)
        
    def reset(self):
        for i in range(5):
            self.node[i].n_reset()
        for i in range(len(self.stdp)):
            self.stdp[i].reset()

class DopamineArea(BrainArea):
    """
    Dopamine brain area with a group of spiking neurons, computes reward prediction error.
    """
    def __init__(self, n_neurons, beta=0.2):
        super().__init__()
        self.n_neurons = n_neurons
        self.beta = beta
        self.node = [IFNode() for _ in range(n_neurons)]
        self.P = np.zeros(n_neurons)  # prediction for each neuron
    def forward(self, spikes):
        out_spikes = []
        for i in range(self.n_neurons):
            spike = self.node[i](torch.tensor([spikes[i]], dtype=torch.float32))
            out_spikes.append(spike)
        S = torch.stack(out_spikes).mean().item()
        delta = S - self.P
        self.P = self.P + self.beta * delta
        return delta, out_spikes
    def reset(self):
        self.P = np.zeros(self.n_neurons)
        for n in self.node:
            n.n_reset()
            
def BAESNN_train():  
    s = env.reset()
    env._set_danger()
    env._set_wall()
    pain = 0
    i = 0
    set_pain = 0
    env._set_switch()
    for i in range(100):
        snn2.reset()
        T = 100
        pain = 0
        print('**************step:', i)
        env.render()
        
        action = np.random.choice(list(range(env.n_actions)))
        print('action:', action)
        d, d_pre, s_, sss = env.step(s, action, pain)
        print('d:', d, 'd_pre:', d_pre, 'sss:', sss)
        env.render()
        
        while (d == np.array([0, 0])).all():
            action = np.random.choice(list(range(env.n_actions)))
            print('action:', action)
            d, d_pre, s_, sss = env.step(s, action, pain)
            print('d:', d, 'd_pre:', d_pre, 'sss:', sss)
            env.render()
        
        # Use env.is_agent1_in_danger to set OUT_PAIN, pain, emotion
        if env.is_agent_in_danger():
            OUT_PAIN = torch.ones(24)
            pain = 1
            set_pain = 1
            emotion = -1
        else:
            OUT_PAIN = torch.zeros(24)
            pain = 0
            emotion = 0

        print("OUT_PAIN:", OUT_PAIN)
        print("pain:", pain)
        print("emotion:", emotion)
        
        T2 = 20
        X1 = OUT_PAIN
        X2 = torch.zeros(24)
        X3 = torch.cat([torch.ones(12) * 0.1, torch.zeros(12)])
        print('X1,X2:', X1, X2)
        spike_emotion = []
        spike_ifg = []
        spike_sma = []
        spike_m1 = []
        spike_per = []
        for i in range(T2):
            if i >= 2:
                X2 = X3
            OUTPUT = snn2(X1, X2)
            snn2.UpdateWeight(2, OUTPUT[0][1], 0.01)
            snn2.UpdateWeight(5, OUTPUT[1][1], -0.1)
            if OUTPUT[2][0] == 1:
                env.canvas.itemconfig(env.rect, fill="red", outline='red')
            if OUTPUT[2][0] == 0:
                env.canvas.itemconfig(env.rect, fill="green", outline='green')
            spike_emotion.append(OUTPUT[5])
            spike_per.append(OUTPUT[6])
            spike_ifg.append(OUTPUT[2])
            spike_sma.append(OUTPUT[3])
            spike_m1.append(OUTPUT[4])
        
        print('out_ifg:', OUTPUT[2])
        print('out_sma:', OUTPUT[3])
        print('out_m1:', OUTPUT[4])
        print('con2:', snn2.connection[2].weight.data)
        print('con5:', snn2.connection[5].weight.data)
        
        spike_emotion = torch.stack(spike_emotion)
        spike_per = torch.stack(spike_per)
        spike_ifg = torch.stack(spike_ifg)
        spike_sma = torch.stack(spike_sma)
        spike_m1 = torch.stack(spike_m1)
        print(spike_emotion.shape)
        env.render()
        
        s = s_
        if set_pain == 1 and pain == 0:
            env.render()
            break
    env.destroy()

def BAESNN_train_alstruism(lamda, E):
    global writer
    for episode in range(E):
        print('*******************episode:', episode, ',factor:', lamda, '*********************************')
        s1,s2=env2.reset()
        env2._set_wall()
        pain1 = 0
        pain2 = 0
        i = 0
        set_pain = 0
        env2.emotion = 0
        env2.empathy_emotion = 0
        env2.empathy_emotion_t_1 = 0
        
        rr = 0
        hh = 0
        g = 0
        a = []
        if episode < 200:
            e_greedy = 0.5
        elif episode < 500:
            e_greedy = 0.7
        elif episode < 700:
            e_greedy = 0.9
        else:
            e_greedy = 1
            
        r = np.random.uniform()
        
        for i in range(100):
            env2.render()
            done = False
            env2.empathy_emotion_t_1=env2.empathy_emotion
            action2 = RL.choose_action(str([(s2[4] + s2[8]) / 2, (s2[5] + s2[9]) / 2,env2.empathy_emotion]),e_greedy=e_greedy)
            s2_, done, done_oval = env2.step2(action2)
            env2.render()
            T=100
            print('**************step:',i)
            env2.render()
            
            action1 = np.random.choice(list(range(env.n_actions)))
            print('action1:',action1)
            if r<= 0.25:
                if i==0:
                    action1=2
            if 0.25<r<=0.5:
                if i==0:
                    action1=1
                if i==1:
                    action1=1
            if 0.5<r<=0.75:
                if i==0:
                    action1=0
                if i==1:
                    action1=1
                if i==2:
                    action1=1
            if 0.75<r<=1.0:
                if i==0:
                    action1=0
                if i==1:
                    action1=3
                if i==2:
                    action1=1
                if i==3:
                    action1=1
            if env2.emotion==0 and set_pain==1:
                pass
            else:     
                d,d_pre,s1_,sss = env2.step1(s1, action1,env2.emotion)
            print('d:',d,'d_pre:',d_pre,'sss:',sss)
            env2.render()
            if env2.is_agent1_in_danger():
                OUT_PAIN = torch.ones(24)
                pain1 = 1
                env2.emotion = -1
                set_pain = 1
            else:
                OUT_PAIN = torch.zeros(24)
                pain1 = 0
                env2.emotion = 0

            print("OUT_PAIN:", OUT_PAIN)
            print("pain1:", pain1)
            print("emotion:", env2.emotion)
            env2.generate_expression1(env2.emotion)
            
            snn2.reset()
            T2 = 20
            X3 = OUT_PAIN.view(1, -1)
            for i in range(T2):
                OUT = snn2.empathy(X3)
                print('out_ifg:', OUT[0])
            
            if OUT[0][0][0] == 1:
                env2.empathy_emotion = -1            
            if OUT[0][0][0] == 0:
                env2.empathy_emotion = 0    
                
            env2.generate_expression2(env2.empathy_emotion)
            env2.render()

            delta,_ = snn1(OUT[0][0])
            reward1 = delta
            _, reward2 = env2.reward2()   
            rr += reward2
                
            RL.learn_A(str([(s2[4] + s2[8]) / 2, (s2[5] + s2[9]) / 2, env2.empathy_emotion_t_1]), action2, env2.lamda * reward1 + reward2, str([(s2_[4] + s2_[8]) / 2, (s2_[5] + s2_[9]) / 2, env2.empathy_emotion]), done_oval)
            
            s1 = s1_
            env2.render()
            if env2.empathy_emotion == 0 and set_pain == 1:
                a.append(i)
            s2 = s2_  
            if done:
                g = 1
                break
        if a != []:
            hh = 1
        helpnumber.append(hh)
        totalreward.append(rr)
        goalnumber.append(g)
        print('totalreward:\n', totalreward)
        print('goalnumber:\n', goalnumber)
        print('helpnumber:\n', helpnumber)
        writer.add_scalar('totalreward', rr, episode)
        writer.add_scalar('helpnumber', hh, episode)

if __name__ == "__main__":
    env = Maze() 
    snn1 = DopamineArea(n_neurons=24, beta=0.2)  
    snn2 = BAESNN() 
    writer = SummaryWriter(log_dir='runs/BAESNN')
    BAESNN_train()
    env.mainloop()
    K=3
    factor=3.0
    Episode=1000
    REWARD=[]
    totalreward=[]
    helpnumber=[]  
    helpave=[]  
    goalnumber=[]
    env2 = Maze2(lamda=factor)
    RL = QLearningTable(actions=list(range(env.n_actions)))
    BAESNN_train_alstruism(factor,Episode)
    helpnumber=np.array(helpnumber)
    for jj in range(Episode//10):
        helpave.append(np.mean(helpnumber[jj*10:(jj+1)*10])*10)
    plt.figure(1,figsize=[18,9])
    axes = plt.gca()
    axes.set_ylim([-30,10])
    plt.plot(totalreward,label=factor)
    plt.legend(loc='lower right')
    plt.title('totalreward')
    plt.savefig('{}-totalreward.jpg'.format(K))
    plt.figure(2,figsize=[18,9])
    axes = plt.gca()
    axes.set_ylim([0,12])
    plt.plot(helpave,label=factor)
    plt.legend(loc='lower right')
    plt.title('helpave')
    plt.savefig('{}-helpave.jpg'.format(K))
    env.mainloop()
    writer.close()
