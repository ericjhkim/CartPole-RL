# Q-learning environment

import model, plots, animate
import random
import numpy as np
import math

# Algorithm duration parameters
box = 1                                         # select box 1 or 2 for discretization
newtons = 50.0                                  # select force with which the cart is moved

simtime = 200                                   # duration of episode (seconds)
num_episodes = 10000                            # number of episodes
number_actions = 2                              # constant force left or right

# Q-learning parameters
learning_rate = 0.5                             # learning rate
epsilon = 1e-4                                  # exploration vs. exploitation (less exploration is favoured)
discount_factor = 0.99                          # discount factor

# Q-learning discretization setup: number of discrete states per state dimension
if box == 1:
    number_states = (4, 7, 3, 3)                # (x,theta, x', theta')
elif box == 2:
    number_states = (4, 13, 3, 3)               # (x,theta, x', theta')

# Initialize Q-table
q_table = np.zeros(number_states + (number_actions,))

# Initialze data collection
score_hist = []

# Variables for stopping condition
best_simtime = 0

for episode in range(num_episodes):

    sim = model.CartPole(box=box,force=newtons) # create new cart pole instance
    state = sim.get_dstates()                   # fetch initial state
    done = False                                # flag to indicate end of episode
    episode_reward = 0                          # accumulate rewards in episode

    while not done and sim.t<simtime:

        # Off-policy: select an action (epsilon-greedy)
        if random.uniform(0,1) < epsilon:       # exploration
            action = np.random.randint(2)
        else:                                   # exploitation
            action = np.argmax(q_table[state])

        # Take action
        force = sim.discrete_actuator_force(action)

        # Advance simulation
        sim.step(force)

        # Observe resulting state
        nxt_state = sim.get_dstates()

        # Observe reward
        reward, done = sim.get_dreward(nxt_state)

        # Find q-value for the best action:
        best_q = np.amax(q_table[nxt_state])

        # Update the Q table
        q_table[state + (action,)] = q_table[state + (action,)] + learning_rate*(reward + discount_factor*best_q - q_table[state + (action,)])
    
        # Update iteration variables
        state = nxt_state
        episode_reward += reward                # update total reward of episode

    score_hist.append(episode_reward)           # store reward
    avg_score = np.mean(score_hist[-100:])
    print('Episode: ',episode,'  ||  Episode reward: ',episode_reward,'  ||  Average score: ',round(avg_score,2))

    # save best sim
    if sim.t_list[-1] > best_simtime:
        best_simtime = sim.t_list[-1]
        best_sim = sim

    # end episodes when survived for full simulation time
    if sim.t >= simtime:
        print('Survival objective reached')
        break

print("Survived for "+str(round(best_sim.t_list[-1],2))+'s')
print(np.rad2deg(np.min(best_sim.theta_list)),np.rad2deg(np.max(best_sim.theta_list)),np.min(best_sim.x_list),np.max(best_sim.x_list))

plots.perf_hist(episode+1,score_hist)
plots.control_sig(best_sim)

animate.animate(best_sim)