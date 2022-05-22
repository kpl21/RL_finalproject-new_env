'''
This is the learner_template.py file that implements 
a RL components for two 2D gridworld environments as 
part of the final project in the COMP4600/5500-Reinforcement Learning 
course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 12/6/2021
'''
import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
import evaluator as ev
import h5py
from large_gridworld_final import Coll as l_col
from small_gridworld_final import Coll as s_col
from large_gridworld_final import animate as l_animate
from small_gridworld_final import animate as s_animate


ENV = 1 #0 for small and 1 for large
# Collision matrix for the small environment
Coll_small = np.array([[0, 0, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 0, 2],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]])

# Collision matrix for the large environment
Coll_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 4, 2, 2, 2, 2, 2, 4, 4, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 4, 4, 2, 2, 2, 4, 4, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 4, 2, 2, 2, 2, 2, 4, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 4, 2, 2, 2, 1, 0, 0, 0, 4, 2, 4, 4, 4, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 2, 2, 2, 1, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 2, 2, 2, 1, 0, 0, 1, 4, 2, 4, 4, 4, 4, 4, 2, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])                

Colls = [Coll_small, Coll_large]
Coll = Colls[ENV]

# Actions
RIGHT = [0, 1]
LEFT = [0, -1]
UP = [-1, 0]
DOWN = [1, 0]
ACTIONS = [UP, DOWN, RIGHT, LEFT]
NA = len(ACTIONS)
ACTION_IDX = [0, 1, 2, 3]
NC, NR = np.shape(Coll)

def transition(s, a):
    '''transition function'''
    # check for stochasticity
    if Coll[s[0], s[1]] == 4:
        a_idx = ACTIONS.index(a)
        p = 0.1*np.ones(NA)
        p[a_idx] = 0.7
        a_idx = np.random.choice(ACTION_IDX, p=p)
        a = ACTIONS[a_idx]        

    sp = [0, 0]
    if (0 <= s[0]+a[0] <= NC-1) and (0 <= s[1]+a[1] <= NR-1):
        if Coll[s[0]+a[0], s[1]+a[1]] != 1:
            sp[0] = s[0]+a[0]
            sp[1] = s[1]+a[1]
            return sp
        else:
            return s
    else:
        return s

def reward(s):
    '''reward function'''
    if Coll[s[0], s[1]] == 3:
        return 0
    elif Coll[s[0], s[1]] == 2:
        return -5.0
    else:
        return -0.1

def random_agent(s0, num_steps=10):
    '''this is a random walker
    your smart algorithm will replace this'''
    s = s0
    T = [s0]
    R = [reward(s)]
    for i in range(num_steps):
        a_idx = np.random.choice(ACTION_IDX)
        a = ACTIONS[a_idx]
        sp = transition(s, a)
        print(sp)
        re = reward(sp)
        R.append(re)
        T.append(sp)
        s = sp
    return T, R

def small_q_all_states():
    states = []

    for height in range(15):
        for width in range(15):
            states.append((height, width))
    return states

def large_q_all_states():
    states = []

    for height in range(30):
        for width in range(30):
            states.append((height, width))
    return states

def max_q(q, state_indice):
    max_q = -100000
    max_action = 0
    pick_option = []
    pick_action = []

    for aa, qq in enumerate(q[state_indice]):

        if qq >= max_q:
            max_q = qq
            max_action = aa
            max_val = (max_q, max_action)
            pick_option.append(max_val)

    for i in range(len(pick_option)):

        if pick_option[i][0] == max_q:
            pick_action.append(pick_option[i][1])

    action = np.random.choice(pick_action)
    return action

def q_epsilon_greedy(q, state_indice, e, num_actions):
    p = np.random.rand()

    if p < e:
        next_action = np.random.choice(range(num_actions))

    else:

        next_action = max_q(q, state_indice)

    return next_action

def small_q_learning(eps):

    num_actions = 4
    e = 0.1
    alpha = 0.1
    states = small_q_all_states()

    start = (14, 2)
    almost_goal = [(1,12), (0,13)]
    terminal = (0,12)
    q = [[0] * num_actions for _ in range(len(states))]
    q_optim = []
    total_reward = []
    timesteps = []
    final_state_history = []
    state_indice = 0
    next_state_indice = 0

    for episodes in range(eps):
        state = start

        reward_hist = []

        t = 0

        while True:

            for index, x in enumerate(states):
                if states[index] == state:
                    state_indice = index
                    break

            action = q_epsilon_greedy(q, state_indice, e, num_actions)
            a_map = ACTIONS[action]
            next_state = tuple(transition(state, a_map))

            for index2, x2 in enumerate(states):
                if states[index2] == next_state:
                    next_state_indice = index2
                    break

            r = reward(state)

            reward_hist.append(r)

            if next_state == terminal:

                q[state_indice][action] += alpha * (r - q[state_indice][action])
                total_reward.append(sum(reward_hist))
                timesteps.append(t)
                print(episodes, t)
                q_optim = q

                if episodes == eps - 1:
                    final_state_history.append(state)
                    if state in almost_goal:
                        final_state_history.append(next_state)

                break

            else:

                next_action = max_q(q, state_indice)

                q[state_indice][action] += alpha * (r + 1 * q[next_state_indice][next_action] - (q[state_indice][action]))

                if episodes == eps - 1:
                    final_state_history.append(state)
                state = next_state
                #print(state)
                t += 1

    return q_optim, final_state_history, timesteps, total_reward

def small_random_q_learning(eps):

    num_actions = 4
    e = 0.01
    alpha = 0.45
    states = small_q_all_states()

    almost_goal = [(1,12), (0,13)]
    terminal = (0,12)
    all_q = [[]] * len(states)

    for j in range(len(all_q)):
        all_q[j] = ([[0] * num_actions for _ in range(len(states))])

    q_optim = [[]] * len(states)
    total_reward = []
    timesteps = []
    final_state_history = []
    state_indice = 0
    next_state_indice = 0

    for start_indice in range(len(states)):

        first_state = states[start_indice]
        print(first_state)

        if s_col[first_state[0]][first_state[1]] == 1:
            q_optim[start_indice] = all_q[start_indice]
            continue

        q = all_q[start_indice]

        for episodes in range(eps):

            state = first_state
            reward_hist = []

            t = 0

            while True:

                for index, x in enumerate(states):
                    if states[index] == state:
                        state_indice = index
                        break

                action = q_epsilon_greedy(q, state_indice, e, num_actions)
                a_map = ACTIONS[action]
                next_state = tuple(transition(state, a_map))

                for index2, x2 in enumerate(states):
                    if states[index2] == next_state:
                        next_state_indice = index2
                        break

                r = reward(state)

                reward_hist.append(r)

                if next_state == terminal:

                    q[state_indice][action] += alpha * (r - q[state_indice][action])
                    total_reward.append(sum(reward_hist))
                    timesteps.append(t)
                    print(episodes, t)
                    q_optim[start_indice] = q

                    if episodes == eps - 1:
                        final_state_history.append(state)
                        if state in almost_goal:
                            final_state_history.append(next_state)

                    break

                else:

                    next_action = max_q(q, state_indice)

                    q[state_indice][action] += alpha * (r + 1 * q[next_state_indice][next_action] - (q[state_indice][action]))

                    if episodes == eps - 1:
                        final_state_history.append(state)
                    state = next_state
                    #print(state)
                    t += 1

    return q_optim, final_state_history, timesteps, total_reward

def large_q_learning(eps):

    num_actions = 4
    e = 0.1
    alpha = 0.8
    states = large_q_all_states()

    start = (0, 0)
    almost_goal = [(13,23), (15,23), (14,24)]
    terminal = (14,23)
    q = [[0] * num_actions for _ in range(len(states))]
    q_optim = []
    total_reward = []
    timesteps = []
    final_state_history = []
    state_indice = 0
    next_state_indice = 0

    for episodes in range(eps):
        state = start

        reward_hist = []

        t = 0

        while True:

            for index, x in enumerate(states):
                if states[index] == state:
                    state_indice = index
                    break

            action = q_epsilon_greedy(q, state_indice, e, num_actions)
            a_map = ACTIONS[action]
            next_state = tuple(transition(state, a_map))

            for index2, x2 in enumerate(states):
                if states[index2] == next_state:
                    next_state_indice = index2
                    break

            r = reward(state)

            reward_hist.append(r)

            if next_state == terminal:

                q[state_indice][action] += alpha * (r - q[state_indice][action])
                total_reward.append(sum(reward_hist))
                timesteps.append(t)
                print(episodes, t)
                q_optim = q

                if episodes == eps - 1:
                    final_state_history.append(state)
                    if state in almost_goal:
                        final_state_history.append(next_state)

                break

            else:

                next_action = max_q(q, state_indice)

                q[state_indice][action] += alpha * (r + 1 * q[next_state_indice][next_action] - (q[state_indice][action]))

                if episodes == eps - 1:
                    final_state_history.append(state)
                state = next_state
                #print(state)
                t += 1

    return q_optim, final_state_history, timesteps, total_reward

def large_random_q_learning(eps):

    num_actions = 4
    e = 0.01
    alpha = 0.8
    states = large_q_all_states()

    almost_goal = [(13,23), (15,23), (14,24)]
    terminal = (14,23)
    all_q = [[]] * len(states)

    for j in range(len(all_q)):
        all_q[j] = ([[0] * num_actions for _ in range(len(states))])

    q_optim = [[]] * len(states)
    total_reward = []
    timesteps = []
    final_state_history = []
    state_indice = 0
    next_state_indice = 0

    for start_indice in range(len(states)):

        first_state = states[start_indice]
        print(first_state)

        if l_col[first_state[0]][first_state[1]] == 1:
            q_optim[start_indice] = all_q[start_indice]
            continue

        if first_state in ev.forbidden_areas:
            q_optim[start_indice] = all_q[start_indice]
            continue

        q = all_q[start_indice]

        for episodes in range(eps):

            state = first_state
            reward_hist = []

            t = 0

            while True:

                for index, x in enumerate(states):
                    if states[index] == state:
                        state_indice = index
                        break

                action = q_epsilon_greedy(q, state_indice, e, num_actions)
                a_map = ACTIONS[action]
                next_state = tuple(transition(state, a_map))

                for index2, x2 in enumerate(states):
                    if states[index2] == next_state:
                        next_state_indice = index2
                        break

                r = reward(state)

                reward_hist.append(r)

                if next_state == terminal:

                    q[state_indice][action] += alpha * (r - q[state_indice][action])
                    total_reward.append(sum(reward_hist))
                    timesteps.append(t)
                    print(episodes, t)
                    q_optim[start_indice] = q

                    if episodes == eps - 1:
                        final_state_history.append(state)
                        if state in almost_goal:
                            final_state_history.append(next_state)

                    break

                else:

                    next_action = max_q(q, state_indice)

                    q[state_indice][action] += alpha * (r + 1 * q[next_state_indice][next_action] - (q[state_indice][action]))

                    if episodes == eps - 1:
                        final_state_history.append(state)
                    state = next_state
                    #print(state)
                    t += 1

    return q_optim, final_state_history, timesteps, total_reward

def small_q_runs(eps, runs):

    q = []
    timesteps = []
    rewards = []

    for i in range(runs):
        result = small_q_learning(eps)
        print(i)

        q.append(result[0])

        if i == runs-1:
            traj = result[1]
        timesteps.append(result[2])
        rewards.append(result[3])

    q = np.mean(np.asarray(q), axis=0)
    print(q)
    return q, traj, timesteps, rewards

def large_q_runs(eps, runs):

    q = []
    timesteps = []
    rewards = []

    for i in range(runs):
        result = large_q_learning(eps)
        print(i)

        q.append(result[0])

        if i == runs-1:
            traj = result[1]
        timesteps.append(result[2])
        rewards.append(result[3])

    q = np.mean(np.asarray(q), axis=0)
    print(q)
    return q, traj, timesteps, rewards

def plot_timesteps(eps, timesteps):
    episodes = []

    for i in range(eps):
        episodes.append(i)

    plt.plot(episodes, np.mean(np.asarray(timesteps), axis=0), label="large-grid: gamma = 1, e = 0.1, alpha = 0.8", color="red")
    #plt.plot(episodes, np.mean(np.asarray(timesteps), axis=0), label="small-grid: gamma = 1, e = 0.1, alpha = 0.1", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Timesteps")
    plt.title("Q-Learning: Timesteps per Episode Over 50 Runs")
    plt.legend()
    plt.show()

def plot_rewards(eps, cum_rewards):
    episodes = []

    for i in range(eps):
        episodes.append(i)

    plt.plot(episodes, np.mean(np.asarray(cum_rewards), axis=0), label="large-grid: gamma = 1, e = 0.1, alpha = 0.8", color="red")
    #plt.plot(episodes, np.mean(np.asarray(cum_rewards), axis=0), label="small-grid: gamma = 1, e = 0.1, alpha = 0.1", color="blue")
    #plt.axhline(y=0, color="blue", linestyle="-", lw=3, label="small grid: maximum reward")
    plt.axhline(y=0, color="red", linestyle="--", lw=3, label="large grid: maximum reward")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Q-Learning: Total Rewards per Episode Over 50 Runs")
    plt.legend()
    plt.show()

if __name__=="__main__":

    # 1000 eps x 50 runs small
    # q1, traj1, timesteps, rewards = small_q_runs(1000, 50)
    # plot_timesteps(1000, timesteps)
    # plot_rewards(1000, rewards)
    # h5f = h5py.File('new_small.h5', 'w')
    # h5f.create_dataset('traj_small', data=traj1)
    # h5f.close()

    # # animate small
    # h5f = h5py.File('new_small.h5', 'r')
    # traj_small = h5f['traj_small'][:]
    # h5f.close()
    # s_animate('trajectory', traj_small)

    # 1000 eps x 50 runs large
    # q2, traj2, timesteps, rewards = large_q_runs(1000, 50)
    # plot_timesteps(1000, timesteps)
    # plot_rewards(1000, rewards)
    # h5f = h5py.File('new_large.h5', 'w')
    # h5f.create_dataset('traj_large', data=traj2)
    # h5f.close()

    # animate large
    h5f = h5py.File('new_large.h5', 'r')
    traj_large = h5f['traj_large'][:]
    h5f.close()
    l_animate('trajectory', traj_large)

    # random initial states: large
    # q_optim2, final_state_history2, timesteps2, total_reward2 = large_random_q_learning(1000)
    # h5f = h5py.File('data.h5', 'w')
    # h5f.create_dataset('op_q2', data=q_optim2)
    # h5f.close()

    # random initial states: small
    # q_optim1, final_state_history1, timesteps1, total_reward1 = small_random_q_learning(4000)
    # q_optim1 = np.asarray(q_optim1)
    # h5f = h5py.File('op_q_small', 'w')
    # h5f.create_dataset('op_q1', data=q_optim1)
    # h5f.close()

    # random initial states: large
    # q_optim2, final_state_history2, timesteps2, total_reward2 = large_random_q_learning(1000)
    # q_optim2 = np.asarray(q_optim2)
    # h5f = h5py.File('op_q_large', 'w')
    # h5f.create_dataset('op_q2', data=q_optim2)
    # h5f.close()