import numpy as np
import matplotlib.pyplot as plt
import learner_template as lt
from sklearn.metrics import mean_squared_error as mse
import small_gridworld_final as sgf
import large_gridworld_final as lgf
from ast import literal_eval as make_tuple
import h5py

# Actions
RIGHT = [0, 1]
LEFT = [0, -1]
UP = [-1, 0]
DOWN = [1, 0]
ACTIONS = [UP, DOWN, RIGHT, LEFT]

forbidden_areas = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5),
                   (23, 3), (23, 4), (23, 5), (24, 3), (24, 4), (24, 5), (25, 3), (25, 4), (25, 5),
                   (3, 23), (3, 24), (3, 25), (4, 23), (4, 24), (4, 25), (5, 23), (5, 24), (5, 25),
                   (23, 23), (23, 24), (23, 25), (24, 23), (24, 24), (24, 25), (25, 23), (25, 24), (25, 25)]

def small_pick_state():
    all_states = lt.small_q_all_states()
    matrix = sgf.Coll
    i = np.random.choice(len(all_states))
    s = all_states[i]

    #(y,x)
    map_id = matrix[s[0]][s[1]]
    if map_id == 1:
        while True:
            i = np.random.choice(len(all_states))
            s = all_states[i]
            map_id = matrix[s[0]][s[1]]
            if map_id != 1:
                break

    return s


def large_pick_state():
    all_states = lt.large_q_all_states()
    matrix = lgf.Coll
    new_matrix = []
    for t in all_states:
        if t not in forbidden_areas:
            new_matrix.append(t)

    i = np.random.choice(len(new_matrix))
    tuple = new_matrix[i]

    if matrix[tuple[0]][tuple[1]] == 1:
        while True:
            i = np.random.choice(len(new_matrix))
            tuple = new_matrix[i]
            if matrix[tuple[0]][tuple[1]] != 1:
                break

    return tuple

def small_eval(q_star, random_state):

    state = random_state
    terminal = (0,12)
    state_history = [state]
    reward_history = []
    states = lt.small_q_all_states()

    for i,s in enumerate(states):
        if state == s:
            start_indice = i

    q = q_star[start_indice]

    state_indice = start_indice

    while state != terminal:

        for index, x in enumerate(states):
            if states[index] == state:
                state_indice = index
                break

        action = lt.max_q(q, state_indice)
        a_map = ACTIONS[action]
        next_s = tuple(lt.transition(state, a_map))
        r = lt.reward(state)
        reward_history.append(r)
        state_history.append(next_s)
        state = next_s
        print("s", state)

    r_sum = sum(reward_history)

    return state_history, r_sum

def large_eval(q_star, random_state):

    state = random_state
    terminal = (14,23)
    state_history = [state]
    reward_history = []
    states = lt.large_q_all_states()

    for i, s in enumerate(states):
        if state == s:
            start_indice = i

    q = q_star[start_indice]

    state_indice = start_indice

    while state != terminal:

        for index, x in enumerate(states):
            if states[index] == state:
                state_indice = index
                break

        action = lt.max_q(q, state_indice)
        a_map = ACTIONS[action]
        next_s = tuple(lt.transition(state, a_map))

        if len(state_history) > 100:
            null_state = [0]
            state_history = null_state
            break

        r = lt.reward(state)
        reward_history.append(r)
        state_history.append(next_s)
        state = next_s
        print("l", state)

    r_sum = sum(reward_history)

    return state_history, r_sum

def random_runs_large(runs, q_star):

    picked_states = []
    rewards = []

    i = 0
    while True:
        random_state = large_pick_state()

        if random_state in picked_states:
            while True:
                random_state = large_pick_state()
                if random_state not in picked_states:
                    break
        picked_states.append(random_state)
        state_history, r_sum = large_eval(q_star, random_state)
        print(i)
        if state_history[0] != 0:
            rewards.append(r_sum)
            i += 1
        if i == runs:
            break


    return rewards


def random_runs_small(runs, q_star):
    picked_states = []
    rewards = []

    for i in range(runs):
        random_state = small_pick_state()

        if random_state not in picked_states:
            while True:
                random_state = small_pick_state()
                if random_state not in picked_states:
                    break
        picked_states.append(random_state)
        state_history, r_sum = small_eval(q_star, random_state)
        rewards.append(r_sum)

    return rewards

def std_dev(reward_hist):

    max_reward = np.zeros(len(reward_hist))
    x = sum(np.abs(reward_hist-max_reward)**2)
    std = (x/(len(reward_hist)-1))**0.5

    return std

def error(r1, r2):

    max_reward = np.zeros(len(r1))
    small_grid_error = np.abs(max_reward - r1)/100
    large_grid_error = np.abs(max_reward - r2)/100

    max_small_grid_error = np.amax(small_grid_error)
    min_small_grid_error = np.amin(small_grid_error)

    max_large_grid_error = np.amax(large_grid_error)
    min_large_grid_error = np.amin(large_grid_error)

    return min_small_grid_error, max_small_grid_error, min_large_grid_error, max_large_grid_error

def random_plot(runs, r2):

    r = []

    for i in range(runs):
        r.append(i)

    plt.plot(r, r2, label="large-grid: gamma = 1, e = 0.01, alpha = 0.8", color="red", marker=".", linestyle="None")
    #plt.plot(r, r1, label="small-grid: gamma = 1, e = 0.01, alpha = 0.45", color="blue", marker=".", linestyle="None")
    #plt.axhline(y=0, color="blue", linestyle="-", lw=3, label="small grid: maximum reward")
    plt.axhline(y=0, color="red", linestyle="--", lw=3, label="large grid: maximum reward")
    plt.xlabel("Run")
    plt.ylabel("Reward")
    plt.title("Q-Learning: Total Rewards per Run Over 100 Random States")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    h5f2 = h5py.File('op_q_large', 'r')
    q_large = h5f2['op_q2'][:]
    q_large = list(q_large)
    #h5f1 = h5py.File('op_q_small', 'r')
    #q_small = h5f1['op_q1'][:]
    #q_small = list(q_small)
    #h5f1.close()
    h5f2.close()

    #r1 = random_runs_small(100, q_small)
    #print(r1)
    r1 = [-1.3, -2.400000000000001, -0.7999999999999999, -1.4000000000000001, -1.6000000000000003, -1.3, -6.399999999999995, -0.1, 0, -0.9999999999999999,
          -0.9999999999999999, -1.3, -1.2, -7.3999999999999915, -1.8000000000000005, -1.5000000000000002, -1.4000000000000001, -0.1, -2.400000000000001, -1.5000000000000002,
          -1.0999999999999999, -0.7, -2.1000000000000005, -1.6000000000000003, -2.2000000000000006, -2.400000000000001, -1.6000000000000003, -2.400000000000001, -2.2000000000000006, -2.700000000000001,
          -6.599999999999994, -2.400000000000001, -1.7000000000000004, -1.2, -6.399999999999995, -2.1000000000000005, -0.1, -2.500000000000001, -6.899999999999993, -0.5,
          -2.0000000000000004, -2.400000000000001, -0.9999999999999999, -1.9000000000000006, -1.3, -1.5000000000000002, -2.600000000000001, -1.9000000000000006, -2.2000000000000006, -2.0000000000000004,
          -0.8999999999999999, -2.3000000000000007, -2.2000000000000006, -1.6000000000000003, -0.1, -7.499999999999991, -1.2, -5.599999999999998, -1.0999999999999999, -2.1000000000000005,
          -1.4000000000000001, -1.5000000000000002, -1.4000000000000001, -2.2000000000000006, -1.4000000000000001, -7.3999999999999915, -1.5000000000000002, -2.500000000000001, -1.6000000000000003, -1.2,
          -1.9000000000000006, -7.3999999999999915, -1.4000000000000001, -1.9000000000000006, -1.7000000000000004, -2.1000000000000005, -5.499999999999998, -1.7000000000000004, -1.6000000000000003, -2.600000000000001,
          -0.7, -2.1000000000000005, -6.899999999999993, -1.6000000000000003, -1.6000000000000003, -6.099999999999996, -0.6, -5.599999999999998, -0.9999999999999999, -0.5,
          -1.5000000000000002, -1.4000000000000001, -6.799999999999994, -0.9999999999999999, -1.4000000000000001, -0.6, -2.0000000000000004, -1.2, -1.4000000000000001, -1.0999999999999999]
    len(r1)
    r2 = random_runs_large(100, q_large)
    print(r2)
    #random_plot(100, r1)
    random_plot(100, r2)

    std_small = 2.9358632340919626
    std_small = std_dev(r1)
    std_large = std_dev(r2)
    print("Small Grid Standard Deviation =", std_small)
    print("Large Grid Standard Deviation =", std_large)

    min_small_grid_error, max_small_grid_error, min_large_grid_error, max_large_grid_error = error(r1, r2)
    print("Min Small Grid Error =", min_small_grid_error)
    print("Max Small Grid Error =", max_small_grid_error)
    print("Min Large Grid Error =", min_large_grid_error)
    print("Max Large Grid Error =", max_large_grid_error)

    max_reward = np.zeros(len(r1))
    small_grid_mse = mse(max_reward,r1)
    large_grid_mse = mse(max_reward, r2)
    print(small_grid_mse, large_grid_mse)
    print("Small Grid MSE =", small_grid_mse)
    print("Large Grid MSE =", large_grid_mse)