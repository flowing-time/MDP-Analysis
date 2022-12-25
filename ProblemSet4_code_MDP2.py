# %%
import numpy as np
import matplotlib.pyplot as plt
from hiive.mdptoolbox import mdp, example
from gym.envs.toy_text.frozen_lake import generate_random_map

map_4by4 = [
            'SFFH',
            'FHFF',
            'FFHF',
            'FFFG'
            ]
map_8by8 = ['SFHFFFFF',
            'HFFFFHFH',
            'FFHFFFFF',
            'FFFFFFFF',
            'FHFFFHHH',
            'HFFFFFFH',
            'HFFFFFFH',
            'FHHHFFFG']



debug = False

def updateR(R, run_map):

    d_to_a = {(-1, 0):0, (0, 1):1, (1, 0):2, (0, -1):3}

    size = len(run_map)
    for i in range(size):
        for j in range(size):
            if run_map[i][j] == 'H':
                for nbh in ((i-1, j), (i+1, j), (i, j-1), (i, j+ 1)):
                    y, x = nbh
                    if 0 <= x < size and 0 <= y < size:
                        keep_away = (x-j, y-i)
                        best_a = d_to_a[keep_away]

                        nbh_state = y * size + x
                        R[nbh_state] += [(-0.05 if a != best_a else 0) for a in range(4)]


def updateR_2(R, run_map):

    size = len(run_map)
    for i in range(size):
        for j in range(size):
            if run_map[i][j] == 'H':
                state = i * size + j
                R[state] += -1



def get_hole_and_goal(run_map):

    result = []
    size = len(run_map)
    for i in range(size):
        for j in range(size):
            if run_map[i][j] in ('H', 'G'):
                state = i * size + j
                result.append(state)
    
    return result




# %%
def policy_visualize(run_map, policy, title, debug=True):

    size = len(run_map)
    d_map_convert = {'S':2, 'F':1, 'H':0, 'G':3}
    d_direction = {0:'<', 1:'V', 2:'>', 3:'^'}

    digi_map = np.zeros((size, size), dtype='int')
    for i in range(size):
        for j in range(size):
            digi_map[i, j] = d_map_convert[run_map[i][j]]

    plt.figure()
    plt.imshow(digi_map)
    plt.axis('off')
    plt.title(title)

    for i in range(size):
        for j in range(size):
            if digi_map[i, j] in (0, 3):
                continue
            plt.text(j, i, d_direction[policy[size*i+j]], color='w')

    #plt.text(0, 0, 'S', color='b')
    plt.text(size-1, size-1, 'G', color='k')

    if debug:
        plt.show()
    else:
        plt.savefig(f'images/MDP2_{title}.png')


# %% Define the problem
#random_map = generate_random_map(size=8, p=0.7)
state_size = '8x8'
random_map = map_8by8
P, R = example.openai("FrozenLake-v1", desc=random_map, is_slippery=True)
#updateR(R, random_map)
#updateR_2(R, random_map)

# ---------------VI------------------------
# %% ValueIteration
vi = mdp.ValueIteration(P, R, 0.99)
vi.run()

#plt.figure()
#plt.plot(vi.policy, '-o')
#plt.xlabel('State')
#plt.ylabel('Action')
#plt.title(f'State {state_size} VI Optimum Policy - Action vs State')
#if debug:
#    plt.show()
#else:
#    plt.savefig(f'images/MDP2_VI_forest_policy{state_size}.png')
policy_visualize(random_map, vi.policy, f'VI_policy_{state_size}', debug=debug)

# %%
plt.figure()
plt.plot([ d['Error'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} VI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_VI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} VI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_VI_Time_Iteration_{state_size}.png')


# ---------------PI------------------------
# %% PolicyIteration
pi = mdp.PolicyIteration(P, R, 0.99, eval_type=1)
pi.run()

# %%
#plt.figure()
#plt.plot(pi.policy, '-o')
#plt.xlabel('State')
#plt.ylabel('Action')
#plt.title(f'State {state_size} PI Optimum Policy - Action vs State')
#if debug:
#    plt.show()
#else:
#    plt.savefig(f'images/MDP2_PI_forest_policy_{state_size}.png')
policy_visualize(random_map, pi.policy, f'PI_policy_{state_size}', debug=debug)

# %%
plt.figure()
plt.plot([ d['Error'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} PI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_PI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} PI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_PI_Time_Iteration_{state_size}.png')


# ---------------QL------------------------
# %%
hole_and_goal_states = get_hole_and_goal(random_map)

def check_if_new_episode1(old_s, action, new_s):
    if new_s in hole_and_goal_states:
        return True
    else:
        return False

ql = mdp.QLearning(P, R, 0.99, epsilon_decay=1-1e-5, alpha=0.2, alpha_decay=1, n_iter=1000000, iter_callback=check_if_new_episode1)
ql.run()
#plt.plot(ql.policy, '-o')


## %%
#plt.figure()
#plt.plot(ql.policy, '-o')
#plt.xlabel('State')
#plt.ylabel('Action')
#plt.title(f'State {state_size} QL Optimum Policy - Action vs State')
#if debug:
#    plt.show()
#else:
#    plt.savefig(f'images/MDP2_QL_forest_policy_{state_size}.png')
policy_visualize(random_map, ql.policy, f'QL_policy_{state_size}', debug=debug)

# %%
plt.figure()
plt.plot([ d['Mean V'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Mean V')
plt.title(f'State {state_size} QL - Mean V vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_QL_Value_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} QL - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_QL_Time_Iteration_{state_size}.png')





# -------------------------------------------
# %% Define the problem
random_map = generate_random_map(size=30, p=0.8)
state_size = '30x30'
#random_map = map_4by4
P, R = example.openai("FrozenLake-v1", desc=random_map, is_slippery=True)
#updateR(R, random_map)
#updateR_2(R, random_map)

# ---------------VI------------------------
# %% ValueIteration
vi = mdp.ValueIteration(P, R, 0.99, epsilon=0.0001)
vi.run()

#plt.figure()
#plt.plot(vi.policy, '-o')
#plt.xlabel('State')
#plt.ylabel('Action')
#plt.title(f'State {state_size} VI Optimum Policy - Action vs State')
#if debug:
#    plt.show()
#else:
#    plt.savefig(f'images/MDP2_VI_forest_policy{state_size}.png')
policy_visualize(random_map, vi.policy, f'VI_policy_{state_size}', debug=debug)

# %%
plt.figure()
plt.plot([ d['Error'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} VI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_VI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} VI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_VI_Time_Iteration_{state_size}.png')


# ---------------PI------------------------
# %% PolicyIteration
pi = mdp.PolicyIteration(P, R, 0.99, eval_type=1)
pi.run()

# %%
#plt.figure()
#plt.plot(pi.policy, '-o')
#plt.xlabel('State')
#plt.ylabel('Action')
#plt.title(f'State {state_size} PI Optimum Policy - Action vs State')
#if debug:
#    plt.show()
#else:
#    plt.savefig(f'images/MDP2_PI_forest_policy_{state_size}.png')
policy_visualize(random_map, pi.policy, f'PI_policy_{state_size}', debug=debug)

# %%
plt.figure()
plt.plot([ d['Error'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} PI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_PI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} PI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_PI_Time_Iteration_{state_size}.png')


# ---------------QL------------------------
# %%
hole_and_goal_states = get_hole_and_goal(random_map)

def check_if_new_episode2(old_s, action, new_s):
    if new_s in hole_and_goal_states:
        return True
    else:
        return False
ql = mdp.QLearning(P, R, 0.99, epsilon_decay=1-1e-5, alpha=0.2, alpha_decay=1, n_iter=5000000, iter_callback=check_if_new_episode2)
#ql = mdp.QLearning(P, R, 0.99, epsilon_decay=1-1e-5, alpha=0.2, alpha_decay=1, n_iter=5000000)
ql.run()
#plt.plot(ql.policy, '-o')


## %%
#plt.figure()
#plt.plot(ql.policy, '-o')
#plt.xlabel('State')
#plt.ylabel('Action')
#plt.title(f'State {state_size} QL Optimum Policy - Action vs State')
#if debug:
#    plt.show()
#else:
#    plt.savefig(f'images/MDP2_QL_forest_policy_{state_size}.png')
policy_visualize(random_map, ql.policy, f'QL_policy_{state_size}', debug=debug)

# %%
plt.figure()
plt.plot([ d['Mean V'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Mean V')
plt.title(f'State {state_size} QL - Mean V vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_QL_Value_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} QL - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP2_QL_Time_Iteration_{state_size}.png')