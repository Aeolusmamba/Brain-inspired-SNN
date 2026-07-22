import numpy as np
np.random.seed(1)
import tkinter as tk
import time
from PIL import ImageGrab

UNIT = 40   # pixels
MAZE_H = 11  # grid horizontal
MAZE_W = 5 # grid vertical

class Snowdrift(tk.Tk, object):
    def __init__(self, n_agents=3, n_snowdrifts=4):
        super(Snowdrift, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right', 'clean'] 
        self.n_actions = len(self.action_space)
        self.n_agents = n_agents
        self.n_snowdrifts = n_snowdrifts
        self.UNIT = 40
        self.MAZE_H = 8
        self.MAZE_W = 8
        self.title('Snowdrift Game')
        self.geometry('{0}x{1}'.format(self.MAZE_H * self.UNIT, self.MAZE_W * self.UNIT)) # canvas
        self.agents = []
        self.agents_pos = []
        self.agents_emotion = [-1] * n_agents 
        self.snowdrifts = []
        self.snowdrifts_pos = []
        self.cleaned = []  
        self.empathy_emotion = 0
        self.empathy_emotion_t_1 = 0
        self.help_signal = 0
        self.lamda = 1.0  
        
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                              height=self.MAZE_H * self.UNIT,
                              width=self.MAZE_W * self.UNIT)

        # Create agents
        colors = ['red', 'blue', 'green']
        for i in range(self.n_agents):
            pos = np.array([np.random.randint(0, self.MAZE_W) * self.UNIT + self.UNIT/2,
                          np.random.randint(0, self.MAZE_H) * self.UNIT + self.UNIT/2])
            agent = self.canvas.create_oval(
                pos[0] - 15, pos[1] - 15,
                pos[0] + 15, pos[1] + 15,
                fill=colors[i])
            self.agents.append(agent)
            self.agents_pos.append(pos)
            if self.agents_emotion[i] == -1:
                self.canvas.itemconfig(agent, fill='gray')

        # Create snowdrifts
        for _ in range(self.n_snowdrifts):
            pos = np.array([np.random.randint(0, self.MAZE_W) * self.UNIT + self.UNIT/2,
                          np.random.randint(0, self.MAZE_H) * self.UNIT + self.UNIT/2])
            points = [
                pos[0], pos[1] - 15,  
                pos[0] - 15, pos[1] + 15,  
                pos[0] + 15, pos[1] + 15   
            ]
            snowdrift = self.canvas.create_polygon(points, fill='black')
            self.snowdrifts.append(snowdrift)
            self.snowdrifts_pos.append(pos)
            self.cleaned.append(False)

        self.canvas.pack()

    def reset(self, agent_id):
        """Reset environment and return initial state index"""
        self.update()
        time.sleep(0.001)
        
        # Reset all states
        self.empathy_emotion = 0
        self.empathy_emotion_t_1 = 0
        self.help_signal = 0
        
        # Reset agents
        for i in range(self.n_agents):
            self.canvas.delete(self.agents[i])
            pos = np.array([np.random.randint(0, self.MAZE_W) * self.UNIT + self.UNIT/2,
                          np.random.randint(0, self.MAZE_H) * self.UNIT + self.UNIT/2])
            self.agents_pos[i] = pos
            self.agents[i] = self.canvas.create_oval(
                pos[0] - 15, pos[1] - 15,
                pos[0] + 15, pos[1] + 15,
                fill='gray') # ['red', 'blue', 'green'][i]
            self.agents_emotion[i] = -1

        # Reset snowdrifts
        for i in range(self.n_snowdrifts):
            if hasattr(self, 'snowdrifts') and len(self.snowdrifts) > i:
                self.canvas.delete(self.snowdrifts[i])
            pos = self.snowdrifts_pos[i]
            points = [
                pos[0], pos[1] - 15,  
                pos[0] - 15, pos[1] + 15,  
                pos[0] + 15, pos[1] + 15   
            ]
            snowdrift = self.canvas.create_polygon(points, fill='black')
            if not hasattr(self, 'snowdrifts') or len(self.snowdrifts) <= i:
                self.snowdrifts.append(snowdrift)
            else:
                self.snowdrifts[i] = snowdrift
        self.cleaned = [False] * self.n_snowdrifts
        
        # Calculate initial state index
        init_state = self._get_state_index(agent_id)
        return init_state

    def step_all(self, actions):
        """Multi-agent environment step
        
        Args:
            actions: List[int] - List of actions for each agent
        Returns:
            next_states: List[int] - Next state index for each agent
            rewards: List[float] - Rewards obtained by each agent
            done: bool - Whether the episode is finished
            info: dict - Additional information
        """
        rewards = [0] * self.n_agents
        empathtrewards_t = [0] * self.n_agents
        next_states = []
        cleaned_this_step = []  # Record snowdrifts cleaned in this step

        # 1. Move phase - all agents move simultaneously
        for agent_id, action in enumerate(actions):
            s = self.agents_pos[agent_id]
            base_action = np.array([0, 0])
            
            if action < 4:  # Move actions
                if action == 0:   # up
                    if s[1] > self.UNIT:
                        base_action[1] -= self.UNIT
                elif action == 1:   # down
                    if s[1] < (self.MAZE_H - 1) * self.UNIT:
                        base_action[1] += self.UNIT
                elif action == 2:   # right
                    if s[0] < (self.MAZE_W - 1) * self.UNIT:
                        base_action[0] += self.UNIT
                elif action == 3:   # left
                    if s[0] > self.UNIT:
                        base_action[0] -= self.UNIT
                        
                self.canvas.move(self.agents[agent_id], base_action[0], base_action[1])
                self.agents_pos[agent_id] = self.agents_pos[agent_id] + base_action

        # 2. Cleaning phase - handle all cleaning actions
        for agent_id, action in enumerate(actions):
            if action == 4:  
                s = self.agents_pos[agent_id]
                for i, pos in enumerate(self.snowdrifts_pos):
                    if all(s == pos) and not self.cleaned[i] and i not in cleaned_this_step:
                        self.canvas.itemconfig(self.snowdrifts[i], fill='') 
                        self.cleaned[i] = True
                        cleaned_this_step.append(agent_id)
                        rewards[agent_id] += 2
                        self.agents_emotion[agent_id] = -1
                        self.canvas.itemconfig(self.agents[agent_id], fill='gray')
                        for j in range(self.n_agents):
                            if j != agent_id:
                                rewards[j] += 6 
                                if self.agents_emotion[j] == -1:
                                    self.agents_emotion[j] = 0
                                    self.canvas.itemconfig(self.agents[j], fill=['red', 'blue', 'green'][j])
                                    empathtrewards_t[agent_id] += 6 

        # 3. Calculate next state for each agent
        for agent_id in range(self.n_agents):
            next_state = self._get_state_index(agent_id)
            next_states.append(next_state)
            empathtrewards_t[agent_id]

        # 4. Check if finished
        done = all(self.cleaned)
        
        info = {
            'cleaned_positions': cleaned_this_step,
            'agent_emotions': self.agents_emotion.copy()
        }
        
        return next_states, rewards, empathtrewards_t, done, info

    def _get_state_index(self, agent_id):
        """Convert state to index value"""
        state_index = 0
        pos = self.agents_pos[agent_id]
        x = int(pos[0] / (self.MAZE_W * self.UNIT) * 8)
        y = int(pos[1] / (self.MAZE_H * self.UNIT) * 8) 
        state_index += x + y * 8
        return state_index

    def render(self):
        """Render environment"""
        time.sleep(0.00001)
        self.update()

