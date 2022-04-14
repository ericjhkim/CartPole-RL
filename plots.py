# Plotting data for analysis

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.grid'] = True

def basic_data(sim):

    fig, axs = plt.subplots(2,2)
    plt.subplots_adjust(hspace = 0.5)
    plt.subplots_adjust(wspace = 0.3)
    fig.suptitle('Simulation Data')

    t = sim.t_list

    axs[0,0].set_title('Position (x)')
    axs[0,0].plot(t, sim.x_list)
    axs[0,0].plot(t, sim.x_max*np.ones(len(sim.x_list)),'r',alpha=0.5)
    axs[0,0].plot(t, -sim.x_max*np.ones(len(sim.x_list)),'r',alpha=0.5)
    axs[0,0].set_ylabel('[m]')
    axs[0,0].set_xlabel('[s]')

    axs[0,1].set_title('Pole angle ('u'\u03b8)')
    axs[0,1].plot(t, np.rad2deg(sim.theta_list))
    axs[0,1].plot(t, np.rad2deg(sim.theta_max)*np.ones(len(sim.theta_list)),'r',alpha=0.5)
    axs[0,1].plot(t, np.rad2deg(-sim.theta_max)*np.ones(len(sim.theta_list)),'r',alpha=0.5)
    axs[0,1].set_ylabel('[degrees]')
    axs[0,1].set_xlabel('[s]')

    axs[1,0].set_title('Velocity (dx)')
    axs[1,0].plot(t, sim.dx_list)
    axs[1,0].set_ylabel('[m/s]')
    axs[1,0].set_xlabel('[s]')

    axs[1,1].set_title('Angular velocity (d'u'\u03b8)')
    axs[1,1].plot(t, np.rad2deg(sim.dtheta_list))
    axs[1,1].set_ylabel('[deg/s]')
    axs[1,1].set_xlabel('[s]')

    plt.show()

def learned_params(sim):
    fig, axs = plt.subplots(2,2)
    plt.subplots_adjust(hspace = 0.5)
    plt.subplots_adjust(wspace = 0.3)
    fig.suptitle('Learned parms')

    axs[0,0].set_title('best_q')
    axs[0,0].plot(sim.best_q_array)

    axs[0,1].set_title('total_rewards')
    axs[0,1].plot(sim.total_rewards_array)

    plt.show()

def rewards(total_rewards_array):
    fig, axs = plt.subplots(2,2)
    plt.subplots_adjust(hspace = 0.5)
    plt.subplots_adjust(wspace = 0.3)
    fig.suptitle('Learned parms')

    axs[0,0].set_title('best_q')
    axs[0,0].plot(total_rewards_array)

    plt.show()

def perf_hist(num_episodes,score_hist):
    plt.plot(list(range(num_episodes)),score_hist)
    plt.title('Episodic reward history')
    plt.xlabel('Episode')
    plt.show()

def control_sig(sim):
    plt.plot(sim.t_list[1:100],sim.action_list[:99])
    plt.title('Control action of agent')
    plt.xlabel('Time (s)')
    plt.ylabel('Control signal')
    plt.show()