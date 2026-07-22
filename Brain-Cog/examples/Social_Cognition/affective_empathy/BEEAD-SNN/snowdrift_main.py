import time
import datetime
import os
import random
import numpy as np
import torch
from sd_env import Snowdrift
from rsnn import RSNN
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Global parameters
N_action = 5  # up, down, left, right, clean
N_state = 64  # 8*8 grid
C = 50
runtime = 100
trace_decay = 0.8

torch.manual_seed(42)
np.random.seed(42)

def encode(n, e):
    z = torch.zeros(N_state, 100) 
    z[n, :] = 1
    z = z * 0.51
    return z

def aoencode(n, e, env, agent_id):
    z = torch.zeros(N_state, 100)
    z[n, :] = 1
    for i in range(len(env.agents_pos)):
        if i != agent_id:
            other_pos = env.agents_pos[i]
            x = int(other_pos[0] / (env.MAZE_W * env.UNIT) * 8)
            y = int(other_pos[1] / (env.MAZE_H * env.UNIT) * 8)
            other_state_idx = x + y * 8
            z[other_state_idx, :] += 0.3
    for i, snow_pos in enumerate(env.snowdrifts_pos):
        if not env.cleaned[i]:
            x = int(snow_pos[0] / (env.MAZE_W * env.UNIT) * 8)
            y = int(snow_pos[1] / (env.MAZE_H * env.UNIT) * 8)
            snow_state_idx = x + y * 8
            z[snow_state_idx, :] += 0.6
    z = z * 0.51
    return z

def poencode(n, e, env, agent_id):
    """
    Encode state as partially observable representation.
    Args:
        n: state index of current agent position
        e: emotion state
        env: environment object
        agent_id: agent ID
    """
    z = torch.zeros(N_state, 100)
    agent_pos = env.agents_pos[agent_id]
    cur_x = int(agent_pos[0] / (env.MAZE_W * env.UNIT) * 8)
    cur_y = int(agent_pos[1] / (env.MAZE_H * env.UNIT) * 8)
    obs_range = 1  # observable grid range
    z[n, :] = 1
    for i in range(len(env.agents_pos)):
        if i != agent_id:
            other_pos = env.agents_pos[i]
            x = int(other_pos[0] / (env.MAZE_W * env.UNIT) * 8)
            y = int(other_pos[1] / (env.MAZE_H * env.UNIT) * 8)
            if abs(x - cur_x) <= obs_range and abs(y - cur_y) <= obs_range:
                other_state_idx = x + y * 8
                z[other_state_idx, :] += 0.3
    for i, snow_pos in enumerate(env.snowdrifts_pos):
        if not env.cleaned[i]:
            x = int(snow_pos[0] / (env.MAZE_W * env.UNIT) * 8)
            y = int(snow_pos[1] / (env.MAZE_H * env.UNIT) * 8)
            if abs(x - cur_x) <= obs_range and abs(y - cur_y) <= obs_range:
                snow_state_idx = x + y * 8
                z[snow_state_idx, :] += 0.6
    z = z * 0.51
    return z

def chooseAct(Net, input, explore, n, env, agent_id):
    count_group = np.zeros(N_action)
    count_output = np.zeros(N_action * C)
    for i_train in range(runtime):
        out, dw = Net(input[:, i_train])
        Net.weight_trace *= trace_decay
        Net.weight_trace += dw[0]
        count_output = count_output + np.array(out)
        for i in range(N_action):
            count_group[i] = count_output[i*C:(i+1)*C].sum()
    agent_pos = env.agents_pos[agent_id]
    at_snowdrift = False
    for i, snow_pos in enumerate(env.snowdrifts_pos):
        if not env.cleaned[i] and all(agent_pos == snow_pos):
            at_snowdrift = True
            break
    if not at_snowdrift:
        count_group[4] = float('-inf')
    if np.random.uniform() < explore:
        if not at_snowdrift:
            action = np.random.randint(0, 4)
        else:
            if count_group.max() > float('-inf'):
                action = count_group.argmax()
            else:
                action = np.random.randint(0, N_action)
    else:
        if not at_snowdrift:
            action = np.random.randint(0, 4)
        else:
            action = np.random.randint(0, N_action)
    return action, Net, dw[0], 0

def train_model(n_agents, lamdas, episodes):
    # TensorBoard writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('run33obs', f'sd_partobs_a{n_agents}_l{lamdas[0]}{lamdas[1]}{lamdas[2]}_e{episodes}_{current_time}', f'')
    writer = SummaryWriter(log_dir)
    nets = [RSNN(N_state, N_action*C) for _ in range(n_agents)]
    learn_steps = [[] for _ in range(n_agents)]
    weight_marks = [np.zeros((N_state, N_action)) for _ in range(n_agents)]
    update_stops = [0 for _ in range(n_agents)]
    empathy_rewards_t = [0] * n_agents
    total_rewards = [0] * n_agents
    env = Snowdrift(n_agents=n_agents, n_snowdrifts=10)
    episode_cleaned_counts = []
    agent_cleaned_counts = [0] * n_agents

    for episode in range(episodes):
        print(f'Episode: {episode}, Lambda: {lamdas}')
        cleaned_count = 0 
        episode_agent_cleaned_count = [0] * n_agents
        states = []
        emotion_t = [-1] * n_agents
        for i in range(n_agents):
            state = env.reset(i)
            states.append(state)
        episode_rewards = [0 for _ in range(n_agents)]
        episode_total_rewards = [0 for _ in range(n_agents)]
        if episode < 100:
            e_greedy = 0.2
        elif episode < 300:
            e_greedy = 0.5
        elif episode < 900:
            e_greedy = 0.9
        else:
            for i in range(n_agents):
                if update_stops[i] == 0:
                    update_stops[i] = 1
            e_greedy = 1
        for t in range(100):
            emotion_tt = emotion_t.copy()
            emotion_t = env.agents_emotion.copy()
            actions = []
            for i in range(n_agents):
                input_state = poencode(states[i], env.agents_emotion[i], env, i)
                action, nets[i], dw, _ = chooseAct(nets[i], input_state, e_greedy, states[i], env, i)
                actions.append(action)
            next_states, rewards, empathy_rewards, done, info = env.step_all(actions)
            cleaned_count += len(info['cleaned_positions'])
            if 'cleaned_by_agent' in info:
                for snow_idx, agent_idx in info['cleaned_by_agent'].items():
                    episode_agent_cleaned_count[agent_idx] += 1
                    agent_cleaned_counts[agent_idx] += 1
            print(f'intereaction {t} :')
            for i in range(n_agents):
                if env.agents_emotion[i] == emotion_t[i] and env.agents_emotion[i] == emotion_tt[i] and (emotion_tt[0]==0 or emotion_tt[1]==0 or emotion_tt[2]==0):
                    total_rewards[i] = lamdas[i] * (empathy_rewards[i] - empathy_rewards_t[i]) + rewards[i]
                elif env.agents_emotion[0]==-1 and env.agents_emotion[1]==-1 and env.agents_emotion[2]==-1:
                    total_rewards[i] = lamdas[i] * (empathy_rewards[i] - empathy_rewards_t[i]) + rewards[i]
                else:
                    total_rewards[i] = lamdas[i] * empathy_rewards[i] + rewards[i]
            print(f'Actions: {actions}, Rewards: {rewards}, Empathy Rewards: {empathy_rewards} Total Rewards: {total_rewards} emotion: {env.agents_emotion}')
            empathy_rewards_t = empathy_rewards
            env.render()
            for i in range(n_agents):
                if update_stops[i] == 0:
                    nets[i].UpdateWeight(total_rewards[i], actions[i], C, states[i])
            states = next_states
            for i in range(n_agents):
                episode_rewards[i] += rewards[i]
                episode_total_rewards[i] += total_rewards[i]
            if done:
                break
        for i in range(n_agents):
            writer.add_scalar(f'ERewards/Agent_{i+1}', episode_rewards[i], episode)
            writer.add_scalar(f'totalRewards/Agent_{i+1}', episode_total_rewards[i], episode)
            writer.add_scalar(f'Cleaned/Agent_{i+1}', episode_agent_cleaned_count[i], episode)
        for i in range(n_agents):
            learn_steps[i].append(episode_rewards[i])
        # Save weights
        if episode == episodes-1:
            for i in range(n_agents):
                torch.save(nets[i].connection[0].weight.data, f'weight_agent{i}_lambda{lamdas}_episode{episode}.pth')
        episode_cleaned_counts.append(cleaned_count)
        writer.add_scalar('Performance/Cleaned_Snowdrifts', cleaned_count, episode)
        print(f'Cleaned Count: {cleaned_count}')
    writer.close()  
    return learn_steps, episode_cleaned_counts

if __name__ == "__main__":
    n_agents = 3
    n_snowdrifts = 10
    self_factors = [[1.51, 1.51, 1.51]]
    all_learn_steps = []
    all_cleaned_counts = []
    for ii in range(len(self_factors)):
        all_learn_steps.append([[] for _ in range(n_agents)])
        all_cleaned_counts.append([])
    for iii, lamdas in enumerate(self_factors):
        steps, cleaned_counts = train_model(n_agents, lamdas, 1000)
        for i in range(n_agents):
            all_learn_steps[iii][i].extend(steps[i])
        all_cleaned_counts[iii] = cleaned_counts
    # Save plot images
    save_dir = 'results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=[24, 8])
    # Plot reward curves for each agent
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green']
    labels = ['Agent 1', 'Agent 2', 'Agent 3']
    for i in range(n_agents):
        plt.plot(all_learn_steps[0][i], label=labels[i], color=colors[i])
    plt.legend(loc='lower right')
    plt.title('Rewards per Agent')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot number of cleaned snowdrifts
    plt.subplot(1, 3, 2)
    plt.plot(all_cleaned_counts[0], label='Cleaned Snowdrifts', color='black')
    plt.axhline(y=n_snowdrifts, color='r', linestyle='--', label='Total Snowdrifts')
    plt.legend(loc='lower right')
    plt.title('Number of Cleaned Snowdrifts per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.ylim([0, n_snowdrifts + 1])

    # Add total reward curve
    plt.subplot(1, 3, 3)
    total_rewards_per_episode = np.sum(all_learn_steps[0], axis=0)  # Calculate total reward for each episode
    plt.plot(total_rewards_per_episode, label='Total Rewards', color='purple')
    plt.legend(loc='lower right')
    plt.title('Total Rewards of All Agents')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.tight_layout()
    # Save image
    save_path = os.path.join(save_dir, f'training_results_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f'Image saved to: {save_path}')
