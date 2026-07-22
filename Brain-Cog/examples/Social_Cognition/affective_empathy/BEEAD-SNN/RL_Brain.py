import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.q_tableA = pd.DataFrame(columns=self.actions)
       
        self.colorblack=0
        self.coloryellow=0
      
    def choose_action(self, observation,e_greedy):
        self.A_chack_q_table_A(observation)
       
        # action selection
        if np.random.uniform() < e_greedy:
            # choose best action
            state_actionA = self.q_tableA.loc[observation, :]
            
            state_action=state_actionA
         
            state_action= state_action.astype(float)
            # print(state_action)
            
            action = state_action.argmax()
            #print('best action',action)
            
        else:
            action = np.random.choice(self.actions)
            #print('random action',action)
        return action

    def learn_A(self, s, a, r, s_,done_oval):
        self.A_chack_q_table_A(s_)
       
        if done_oval==0:
            q_predict = self.q_tableA.loc[s, a]
            q_target = r + self.gamma * self.q_tableA.loc[s_, :].max()  
            self.q_tableA.loc[s, a] += self.lr * (q_target - q_predict)  # update
            #print('self.q_tableA:\n',self.q_tableA)
        else:
            pass
        
    def A_chack_q_table_A(self, state):
        if state not in self.q_tableA.index:
            # append new state to q table
            # self.q_tableA = self.q_tableA.append( # append方法被新版本的pandas弃用
            #     pd.Series(
            #         [0]*len(self.actions),
            #         index=self.q_tableA.columns,
            #         name=state,
            #     )
            # )
            self.q_tableA = pd.concat([
                self.q_tableA,
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_tableA.columns,
                    name=state,
                ).to_frame().T
            ])
            
class EnvModel:
    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self, actions):
        # the simplest case is to think about the model is a memory which has all past transition information
        self.actions = actions
        self.database = pd.DataFrame(columns=actions, dtype=np.object)

    def store_transition(self, s, a, r, s_):
        if s not in self.database.index:
            # self.database = self.database.append( append方法被新版本的pandas弃用
            #     pd.Series(
            #         [None] * len(self.actions),
            #         index=self.database.columns,
            #         name=s,
            #     ))
            self.database = pd.concat([
                self.database,
                pd.Series(
                    [None] * len(self.actions),
                    index=self.database.columns,
                    name=s,
                ).to_frame().T
            ])
        self.database.set_value(s, a, (r, s_))

    def sample_s_a(self):
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.loc[s].dropna().index)    # filter out the None value
        return s, a

    def get_r_s_(self, s, a):
        r, s_ = self.database.loc[s, a]
        return r, s_
