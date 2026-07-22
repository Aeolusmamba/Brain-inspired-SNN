import numpy as np
np.random.seed(1)
import tkinter as tk
import time
from PIL import ImageGrab

UNIT = 40   # pixels
MAZE_H = 11  # grid height
MAZE_W = 5 # grid width

class Maze2(tk.Tk, object):
    def __init__(self,lamda=0):
        super(Maze2, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_space1 = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_actions1 = len(self.action_space1)
        self.title('pain_empathy')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._build_maze()
        self.action_hurt1 = 0
        self.emotion=0
        self.empathy_emotion=0 
        self.delta=0.5
        self.set_pain=0
        self.help_signal=0
        self.lamda=lamda
        self.empathy_emotion_t_1=0
            
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_W * UNIT,
                           width=MAZE_H * UNIT)
        
        for c in range(0, MAZE_H * UNIT, UNIT):# create grids
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        
        # self.oval_center = np.array([(MAZE_H * UNIT)/2+80, UNIT/2+UNIT])# create switch
        # self.oval = self.canvas.create_oval(
        #     self.oval_center[0] - 15, self.oval_center[1] - 15,
        #     self.oval_center[0] + 15, self.oval_center[1] + 15,
        #     fill='yellow')
        # self.help = self.canvas.coords(self.oval)




        self.orgin1 = np.array([20, 20])
        # 下
        self.points1 = [
            # 左上
            self.orgin1[0] - 15,  # 5
            self.orgin1[1] - 15,  # 5
            # 右上
            self.orgin1[0] + 15,  # 35
            self.orgin1[1] - 15,  # 5
            # 右下+
            self.orgin1[0] + 15,  # 35
            self.orgin1[1],  # 20
            # 顶点
            self.orgin1[0],  # 20
            self.orgin1[1] + 15,  # 35
            # 左下+
            self.orgin1[0] - 15,  # 5
            self.orgin1[1],  # 20
        ]
        self.agent1 = self.canvas.create_polygon(self.points1, outline='black',fill="blue")# left agent




        self.orgin2 = np.array([MAZE_H * UNIT - UNIT / 2, 20])
        # 下
        self.points2 = [
            # 左上
            self.orgin2[0] - 15,  # 5
            self.orgin2[1] - 15,  # 5
            # 右上
            self.orgin2[0] + 15,  # 35
            self.orgin2[1] - 15,  # 5
            # 右下+
            self.orgin2[0] + 15,  # 35
            self.orgin2[1],  # 20
            # 顶点
            self.orgin2[0],  # 20
            self.orgin2[1] + 15,  # 35
            # 左下+
            self.orgin2[0] - 15,  # 5
            self.orgin2[1],  # 20
        ]
        self.agent2 = self.canvas.create_polygon(self.points2, fill="green")# right agent
        
        
        
        
        self.goal_centre = np.array([(MAZE_H/2) * UNIT + UNIT, (MAZE_W/2) * UNIT])
        # 下
        self.points3 = [
            # 左上
            self.goal_centre[0] - 15,  # 5
            self.goal_centre[1] ,  # 5
            # 右上
            self.goal_centre[0] ,  # 35
            self.goal_centre[1] - 15,  # 5
            # 右下+
            self.goal_centre[0] + 15,  # 35
            self.goal_centre[1],  # 20
            # 顶点
            self.goal_centre[0],  # 20
            self.goal_centre[1] + 15,  # 35 
        ]
        self.target = self.canvas.create_polygon(self.points3, fill="purple") # goal
       
        
       
        
        # self.food=self.canvas.create_arc(((MAZE_H-2) * UNIT +10 ,160,(MAZE_H) * UNIT-5 ,220), 
        #                                  start=0, extent=60, fill='red', outline='orange', width=2)#food
       
        
        
        self.hell1_center = np.array([60, 20])
        self.hell1 = self.canvas.create_oval(
            self.hell1_center[0] - 15, self.hell1_center[1] - 15,
            self.hell1_center[0] + 15, self.hell1_center[1] + 15,
            fill='black')
        self.hell2_center = np.array([20, 100])
        self.hell2 = self.canvas.create_oval(
            self.hell2_center[0] - 15, self.hell2_center[1] - 15,
            self.hell2_center[0] + 15, self.hell2_center[1] + 15,
            fill='black')
        self.hell3_center = np.array([140, 140])
        self.hell3 = self.canvas.create_oval(
            self.hell3_center[0] - 15, self.hell3_center[1] - 15,
            self.hell3_center[0] + 15, self.hell3_center[1] + 15,
            fill='black')
        self.hell4_center = np.array([140, 60])
        self.hell4 = self.canvas.create_oval(
            self.hell4_center[0] - 15, self.hell4_center[1] - 15,
            self.hell4_center[0] + 15, self.hell4_center[1] + 15,
            fill='black')
     

        self.canvas.pack()

    #reset agent location
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.help_signal=0
        self.action_hurt1 = 0
        self.empathy_emotion = 0  # 修正拼写
        
        
        self.canvas.delete(self.agent1)
        self.canvas.delete(self.agent2)
        self.orgin1 = np.array([20, 20])
        # 下
        self.points1 = [
            # 左上
            self.orgin1[0] - 15,  # 5
            self.orgin1[1] - 15,  # 5
            # 右上
            self.orgin1[0] + 15,  # 35
            self.orgin1[1] - 15,  # 5
            # 右下+
            self.orgin1[0] + 15,  # 35
            self.orgin1[1],  # 20
            # 顶点
            self.orgin1[0],  # 20
            self.orgin1[1] + 15,  # 35
            # 左下+
            self.orgin1[0] - 15,  # 5
            self.orgin1[1],  # 20
        ]
        self.agent1 = self.canvas.create_polygon(self.points1, outline='black',fill="blue")

        self.orgin2 = np.array([MAZE_H * UNIT - UNIT / 2, 20])
        # 下
        self.points2 = [
            # 左上
            self.orgin2[0] - 15,  # 5
            self.orgin2[1] - 15,  # 5
            # 右上
            self.orgin2[0] + 15,  # 35
            self.orgin2[1] - 15,  # 5
            # 右下+
            self.orgin2[0] + 15,  # 35
            self.orgin2[1],  # 20
            # 顶点
            self.orgin2[0],  # 20
            self.orgin2[1] + 15,  # 35
            # 左下+
            self.orgin2[0] - 15,  # 5
            self.orgin2[1],  # 20
        ]
        self.agent2 = self.canvas.create_polygon(self.points2, fill="green")
        return self.canvas.coords(self.agent1),self.canvas.coords(self.agent2)


    # move agent1
    def step1(self, s1, action1,emotion):
        s1 = self.canvas.coords(self.agent1)
        self.help_signal = 0 
        self.centre1 = [(s1[4] + s1[8]) / 2, (s1[5] + s1[9]) / 2]
        if all(self.centre1 == self.hell1_center):
            self.action_hurt1 = 1
        if all(self.centre1 == self.hell2_center):
            self.action_hurt1 = 1
        if all(self.centre1 == self.hell3_center):
            self.action_hurt1 = 1   
        if all(self.centre1 == self.hell4_center):
            self.action_hurt1 = 1  
        # if self.help_signal:
        #     self.action_hurt1 = 0
        
            
        self.canvas.delete(self.agent1)  
        self.centre1 = [(s1[4] + s1[8]) / 2, (s1[5] + s1[9]) / 2]
        if action1==0:
            self.points0 = [
                # 右下
                self.centre1[0] + 15,  # 35
                self.centre1[1] + 15,  # 35
                # 左下
                self.centre1[0] - 15,  # 5
                self.centre1[1] + 15,  # 35
                # 左上+
                self.centre1[0] - 15,  # 5
                self.centre1[1],  # 20
                # 顶点
                self.centre1[0],  # 20
                self.centre1[1] - 15,  # 5
                # 右上+
                self.centre1[0] + 15,  # 35
                self.centre1[1],  # 20
            ]
            if emotion==0:
                color="blue"
            if emotion == -1:
                color = "red"
            self.agent1 = self.canvas.create_polygon(self.points0, fill=color,outline='black')
        if action1==1:
            self.points1 = [
                # 左上
                self.centre1[0] - 15,  # 5
                self.centre1[1] - 15,  # 5
                # 右上
                self.centre1[0] + 15,  # 35
                self.centre1[1] - 15,  # 5
                # 右下+
                self.centre1[0] + 15,  # 35
                self.centre1[1],  # 20
                # 顶点
                self.centre1[0],  # 20
                self.centre1[1] + 15,  # 35
                # 左下+
                self.centre1[0] - 15,  # 5
                self.centre1[1],  # 20
            ]
            if emotion==0:
                color="blue"
            if emotion == -1:
                color = "red"
            self.agent1 = self.canvas.create_polygon(self.points1, fill=color,outline='black')
        if action1==2:
            self.points2 = [
                # 左下
                self.centre1[0] - 15,  # 5
                self.centre1[1] + 15,  # 35
                # 左上
                self.centre1[0] - 15,  # 5
                self.centre1[1] - 15,  # 5
                # 右上+
                self.centre1[0],  # 20
                self.centre1[1] - 15,  # 5
                # 顶点
                self.centre1[0] + 15,  # 35
                self.centre1[1],  # 20
                # 右下+
                self.centre1[0],  # 20
                self.centre1[1] + 15,  # 35
            ]
            if emotion==0:
                color="blue"
            if emotion == -1:
                color = "red"
            self.agent1 = self.canvas.create_polygon(self.points2, fill=color,outline='black')
        if action1==3:
            self.points3 = [
                # 右上
                self.centre1[0] + 15,  # 20+15
                self.centre1[1] - 15,  # 20-15
                # 右下
                self.centre1[0] + 15,  # 20+15
                self.centre1[1] + 15,  # 20+15
                # 左下+
                self.centre1[0],  # 20
                self.centre1[1] + 15,  # 20+15
                # 顶点
                self.centre1[0] - 15,  # 20-15
                self.centre1[1],  # 20
                # 左上+
                self.centre1[0],  # 20
                self.centre1[1] - 15,  # 20-15
            ]
            if emotion==0:
                color="blue"
            if emotion == -1:
                color = "red"
            self.agent1 = self.canvas.create_polygon(self.points3, fill=color,outline='black')
        s1 = self.canvas.coords(self.agent1)
        self.render()#显示当前的动作指令是什么

        # whether hurt
        if self.action_hurt1 == 0:
            true_action1 = action1
        else:
            if action1 == 0:
                true_action1 = 1
            if action1 == 1:
                true_action1 = 0
            if action1 == 2:
                true_action1 = 3
            if action1 == 3:
                true_action1 = 2
                
                
        self.centre1 = [(s1[4] + s1[8]) / 2, (s1[5] + s1[9]) / 2] 
        
        # predict next state
        pre_displacement1 = np.array([0, 0])
        if self.centre1[0] <= ((MAZE_H - 1) / 2 + 1) * UNIT:  # 120
            if action1 == 0:  # up
                if self.centre1[1] > UNIT:
                    pre_displacement1 = np.array([0, -40])
            elif action1 == 1:  # down
                if self.centre1[1] < (MAZE_W - 1) * UNIT:
                    pre_displacement1 = np.array([0, 40])
            elif action1 == 2:  # right
                if self.centre1[0] < ((MAZE_H - 1) / 2 - 1) * UNIT:
                    pre_displacement1 = np.array([40, 0])
            elif action1 == 3:  # left
                if self.centre1[0] > UNIT:
                    pre_displacement1 = np.array([-40, 0])
        else:
            if action1 == 0:  # up
                if self.centre1[1] > UNIT:
                    pre_displacement1 = np.array([0, -40])
            elif action1 == 1:  # down
                if self.centre1[1] < (MAZE_W - 1) * UNIT:
                    pre_displacement1 = np.array([0, 40])
            elif action1 == 2:  # right
                if self.centre1[0] < (MAZE_H - 1) * UNIT:
                    pre_displacement1 = np.array([40, 0])
            elif action1 == 3:  # left
                if self.centre1[0] > ((MAZE_H - 1) / 2 + 2) * UNIT:
                    pre_displacement1 = np.array([-40, 0])
       
        
        # true next state
        displacement1 = np.array([0, 0])
        
        if self.centre1[0] <= ((MAZE_H - 1) / 2 + 1) * UNIT:
            if true_action1 == 0:  # up
                if self.centre1[1] > UNIT:
                    displacement1= np.array([0, -40])
            elif true_action1 == 1:  # down
                if self.centre1[1] < (MAZE_W - 1) * UNIT:
                    displacement1= np.array([0,40])
            elif true_action1 == 2:  # right
                if self.centre1[0] < ((MAZE_H - 1) / 2 - 1) * UNIT:
                    displacement1= np.array([40,0])
            elif true_action1 == 3:  # left
                if self.centre1[0] > UNIT:
                    displacement1= np.array([-40,0])
        else:
            if true_action1 == 0:  # up
                if self.centre1[1] > UNIT:
                    displacement1= np.array([0,-40])
            elif true_action1 == 1:  # down
                if self.centre1[1] < (MAZE_W - 1) * UNIT:
                    displacement1= np.array([0,40])
            elif true_action1 == 2:  # right
                if self.centre1[0] < (MAZE_H - 1) * UNIT:
                    displacement1= np.array([40,0])
            elif true_action1 == 3:  # left
                if self.centre1[0] > ((MAZE_H - 1) / 2 + 2) * UNIT:
                    displacement1= np.array([-40,0])
        self.canvas.move(self.agent1, displacement1[0], displacement1[1])
        s1_ = self.canvas.coords(self.agent1)
        sss = [(s1_[4] + s1_[8]) / 2, (s1_[5] + s1_[9]) / 2]

        return displacement1, pre_displacement1,s1_,sss
    
    def is_agent1_in_danger(self):
        """
        whether in danger(hell1~hell4)
        :return: True/False
        """
        s1 = self.canvas.coords(self.agent1)
        print([(s1[4] + s1[8]) / 2, (s1[5] + s1[9]) / 2])
        print(f'hell1_center: {self.hell1_center}, hell2_center: {self.hell2_center}, hell3_center: {self.hell3_center}, hell4_center: {self.hell4_center}')
        agent1_pos = [(s1[4] + s1[8]) / 2, (s1[5] + s1[9]) / 2]
        danger_centers = [self.hell1_center, self.hell2_center, self.hell3_center, self.hell4_center]
        for center in danger_centers:
            if all(np.isclose(agent1_pos, center)):
                return True
        return False

    def step2(self, action):
        s = self.canvas.coords(self.agent2)
        
        s=[(s[4] + s[8]) / 2, (s[5] + s[9]) / 2]
        if all(s == self.oval_center)and self.empathy_emotion==-1:
            done_oval=1
        else:
            done_oval=0
       
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_W - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_H - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > ( (MAZE_H - 1)/2+2) * UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.agent2, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.agent2)  # next state
        self.centre2= [(s_[4] + s_[8]) / 2, (s_[5] + s_[9]) / 2]
        
        
        
        if all(self.centre2 == self.oval_center) and self.empathy_emotion==-1:
           
            self.help_signal=1
      
           
        if all(self.centre2 == self.goal_centre):
        
            done = True
        else:
           
            done = False
            
            
            
            
        return s_, done,done_oval


    def reward2(self):
        
        s_ = self.canvas.coords(self.agent2) 
        self.centre2= [(s_[4] + s_[8]) / 2, (s_[5] + s_[9]) / 2]
        
        if (self.empathy_emotion - self.empathy_emotion_t_1)==-1:
            reward1=0
        
        elif (self.empathy_emotion - self.empathy_emotion_t_1)==1:
            reward1=10
        else:
            reward1=0
       
            
        if all(self.centre2 == self.goal_centre):
            reward2 = 10    
        else:
            reward2 = -1
            
               
            
        return  reward1,reward2


    def _set_wall(self):
        
        self.oval_center = np.array([(MAZE_H * UNIT)-20, ((MAZE_W)*UNIT-20)])# [(MAZE_H * UNIT)/2+80, UNIT/2+UNIT]
        self.oval = self.canvas.create_oval(
            self.oval_center[0] - 15, self.oval_center[1] - 15,
            self.oval_center[0] + 15, self.oval_center[1] + 15,
            fill='yellow')
        self.help = self.canvas.coords(self.oval)
        wall_center = []
        self.wall = []
        for i in range(MAZE_W):
            wall_center.append([])
            self.wall.append([])
        for i in range(MAZE_W):
            wall_center[i] = np.array([(MAZE_H * UNIT) / 2, ((i) * UNIT) + UNIT / 2])# wall
            self.wall[i] = self.canvas.create_rectangle(
                    wall_center[i][0] - 20, wall_center[i][1] - 20,
                    wall_center[i][0] + 20, wall_center[i][1] + 20,
                    fill='grey')
        self.canvas.pack()





    def generate_expression1(self,emotion):
        if emotion==-1:
            self.canvas.itemconfig(self.agent1, fill="red", outline='black')
            self.canvas.pack()
        if emotion==0:
            self.canvas.itemconfig(self.agent1, fill="blue", outline='black')
            self.canvas.pack()
 
    
    def generate_expression2(self,emotion):
        if emotion==-1:
            self.canvas.itemconfig(self.agent2, fill="red")
            self.canvas.pack()
        if emotion==0:
            self.canvas.itemconfig(self.agent2, fill="green")
            self.canvas.pack()
 
 
    def render(self):
        time.sleep(0.000001)
        self.update()

    # def getter(self,widget):
    #     widget.update()
    #     x = tk.Tk.winfo_rootx(self) + widget.winfo_x()
    #     y = tk.Tk.winfo_rooty(self) + widget.winfo_y()
    #     x1 = x + widget.winfo_width()
    #     y1 = y + widget.winfo_height()
    #     ImageGrab.grab().crop((x, y, x1, y1)).save("first.jpg")
    #     return ImageGrab.grab().crop((x, y, x1, y1))

