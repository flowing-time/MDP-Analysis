# %%
import matplotlib.pyplot as plt
from hiive.mdptoolbox import mdp, example

debug = False

# %% Define the problem size = 50
#P, R = example.forest(S=50, r1=100)
state_size = '50_4'
P, R = example.forest(S=50)

# %% ValueIteration
vi = mdp.ValueIteration(P, R, 0.99)
vi.run()

# %%
plt.figure()
plt.plot(vi.policy, '-o')
plt.xlabel('State')
plt.ylabel('Action')
plt.title(f'State {state_size} VI Optimum Policy - Action vs State')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_VI_forest_policy{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Error'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} VI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_VI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} VI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_VI_Time_Iteration_{state_size}.png')




# %% PolicyIteration
pi = mdp.PolicyIteration(P, R, 0.99)
pi.run()

# %%
plt.figure()
plt.plot(pi.policy, '-o')
plt.xlabel('State')
plt.ylabel('Action')
plt.title(f'State {state_size} PI Optimum Policy - Action vs State')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_PI_forest_policy_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Error'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} PI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_PI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} PI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_PI_Time_Iteration_{state_size}.png')



# %% Qlearning
#ql = mdp.QLearning(P, R, 0.99, epsilon_decay=1-1e-5, alpha=0.2, alpha_decay=1)
#for episode in range(100):
#    ql.run()
#plt.plot(ql.policy, '-o')

# %%
ql = mdp.QLearning(P, R, 0.99, epsilon_decay=1-1e-5, alpha=0.2, alpha_decay=1, n_iter=2000000)
ql.run()
#plt.plot(ql.policy, '-o')


# %%
plt.figure()
plt.plot(ql.policy, '-o')
plt.xlabel('State')
plt.ylabel('Action')
plt.title(f'State {state_size} QL Optimum Policy - Action vs State')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_QL_forest_policy_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Mean V'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Mean V')
plt.title(f'State {state_size} QL - Mean V vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_QL_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} QL - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_QL_Time_Iteration_{state_size}.png')



# %% Define the problem size = 1000
state_size = '1000_1000'
P, R = example.forest(S=1000, r1=1000)

# %% ValueIteration
vi = mdp.ValueIteration(P, R, 0.99)
vi.run()

# %%
plt.figure()
plt.plot(vi.policy, '-o')
plt.xlabel('State')
plt.ylabel('Action')
plt.title(f'State {state_size} VI Optimum Policy - Action vs State')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_VI_forest_policy{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Error'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} VI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_VI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} VI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_VI_Time_Iteration_{state_size}.png')




# %% PolicyIteration
pi = mdp.PolicyIteration(P, R, 0.99)
pi.run()

# %%
plt.figure()
plt.plot(pi.policy, '-o')
plt.xlabel('State')
plt.ylabel('Action')
plt.title(f'State {state_size} PI Optimum Policy - Action vs State')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_PI_forest_policy_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Error'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} PI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_PI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} PI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_PI_Time_Iteration_{state_size}.png')



# %% Qlearning
#ql = mdp.QLearning(P, R, 0.99, epsilon_decay=1-1e-5, alpha=0.2, alpha_decay=1)
#for episode in range(100):
#    ql.run()
#plt.plot(ql.policy, '-o')

# %%
ql = mdp.QLearning(P, R, 0.99, epsilon_decay=1-1e-5, alpha=0.2, alpha_decay=1, n_iter=10000000)
ql.run()
#plt.plot(ql.policy, '-o')


# %%
plt.figure()
plt.plot(ql.policy, '-o')
plt.xlabel('State')
plt.ylabel('Action')
plt.title(f'State {state_size} QL Optimum Policy - Action vs State')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_QL_forest_policy_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Mean V'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Mean V')
plt.title(f'State {state_size} QL - Mean V vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_QL_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} QL - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_QL_Time_Iteration_{state_size}.png')



# -------------------------------------------------------
# %% Define the problem size = 50
P, R = example.forest(S=50, r1=50)
state_size = '50_50'
#P, R = example.forest(S=state_size)

# %% ValueIteration
vi = mdp.ValueIteration(P, R, 0.99)
vi.run()

# %%
plt.figure()
plt.plot(vi.policy, '-o')
plt.xlabel('State')
plt.ylabel('Action')
plt.title(f'State {state_size} VI Optimum Policy - Action vs State')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_VI_forest_policy{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Error'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} VI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_VI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in vi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} VI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_VI_Time_Iteration_{state_size}.png')




# %% PolicyIteration
pi = mdp.PolicyIteration(P, R, 0.99)
pi.run()

# %%
plt.figure()
plt.plot(pi.policy, '-o')
plt.xlabel('State')
plt.ylabel('Action')
plt.title(f'State {state_size} PI Optimum Policy - Action vs State')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_PI_forest_policy_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Error'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'State {state_size} PI - Error vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_PI_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in pi.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} PI - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_PI_Time_Iteration_{state_size}.png')



# %% Qlearning
#ql = mdp.QLearning(P, R, 0.99, epsilon_decay=1-1e-5, alpha=0.2, alpha_decay=1)
#for episode in range(100):
#    ql.run()
#plt.plot(ql.policy, '-o')

# %%
ql = mdp.QLearning(P, R, 0.99, epsilon_decay=1-1e-5, alpha=0.2, alpha_decay=1, n_iter=5000000)
ql.run()
#plt.plot(ql.policy, '-o')


# %%
plt.figure()
plt.plot(ql.policy, '-o')
plt.xlabel('State')
plt.ylabel('Action')
plt.title(f'State {state_size} QL Optimum Policy - Action vs State')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_QL_forest_policy_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Mean V'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Mean V')
plt.title(f'State {state_size} QL - Mean V vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_QL_Error_convergence_{state_size}.png')

# %%
plt.figure()
plt.plot([ d['Time'] for d in ql.run_stats ])
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.title(f'State {state_size} QL - Time vs Iteration')
if debug:
    plt.show()
else:
    plt.savefig(f'images/MDP1_QL_Time_Iteration_{state_size}.png')