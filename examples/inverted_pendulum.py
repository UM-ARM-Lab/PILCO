import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pilco.controllers import RbfController
from pilco.models import PILCO
from utils import rollout

np.random.seed(0)

with tf.Session(graph=tf.Graph()) as sess:
    env = gym.make('InvertedPendulum-v2')

    # Evaluate random actions so we know how bad random is
    random_rewards = []
    for i in range(1, 100):
        _, Y_, rewards = rollout(env=env, pilco=None, random=True, timesteps=40)
        random_rewards.append(sum(rewards))

    # Initial random rollouts to generate a dataset
    X, Y, _ = rollout(env=env, pilco=None, random=True, timesteps=40)
    random_rewards = []
    for i in range(1, 3):
        X_, Y_, rewards = rollout(env=env, pilco=None, random=True, timesteps=40)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))
        random_rewards.append(sum(rewards))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
    # controller = LinearController(state_dim=state_dim, control_dim=control_dim)

    pilco = PILCO(X, Y, controller=controller, horizon=40)
    # Example of user provided reward function, setting a custom target state
    # R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
    # pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

    # Example of fixing a parameter, optional, for a linear controller only
    # pilco.controller.b = np.array([[0.0]])
    # pilco.controller.b.trainable = False

    for rollouts in range(3):
        pilco.optimize_models()
        pilco.optimize_policy()
        X_new, Y_new, _ = rollout(env=env, pilco=pilco, timesteps=100)

        # Update dataset
        X = np.vstack((X, X_new))
        Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_XY(X, Y)

    input("done training! Press enter to evaluate the policy")
    trained_rewards = []
    for i in range(1, 100):
        X_, Y_, rewards = rollout(env=env, pilco=pilco, timesteps=100)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))
        trained_rewards.append(sum(rewards))

    plt.figure()
    plt.hist(random_rewards, label='random')
    plt.hist(trained_rewards, label='trained')
    plt.xlabel("reward")
    plt.ylabel("count")
    plt.legend()
    plt.show()
