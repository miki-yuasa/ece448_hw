"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
"""
from typing import Final
import random
import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn


class q_learner:
    def __init__(
        self,
        alpha: float,
        epsilon: float,
        gamma: float,
        nfirst: float,
        state_cardinality: list[int],
    ):
        """
        Create a new q_learner object.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a Q table and an N table.
        Q[...state..., ...action...] = expected utility of state/action pair.
        N[...state..., ...action...] = # times state/action has been explored.
        Both are initialized to all zeros.
        Up to you: how will you encode the state and action in order to
        define these two lookup tables?  The state will be a list of 5 integers,
        such that 0 <= state[i] < state_cardinality[i] for 0 <= i < 5.
        The action will be either -1, 0, or 1.
        It is up to you to decide how to convert an input state and action
        into indices that you can use to access your stored Q and N tables.

        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting
        state_cardinality (list) - cardinality of each of the quantized state variables

        @return:
        None
        """
        self._alpha: Final = alpha
        self._epsilon: Final = epsilon
        self._gamma: Final = gamma
        self._nfirst: Final = nfirst
        self._state_cardinality: Final = state_cardinality
        self.Q: ndarray = np.zeros(
            (
                state_cardinality[0],
                state_cardinality[1],
                state_cardinality[2],
                state_cardinality[3],
                state_cardinality[4],
                3,
            )
        )
        self.N: ndarray = np.zeros(
            (
                state_cardinality[0],
                state_cardinality[1],
                state_cardinality[2],
                state_cardinality[3],
                state_cardinality[4],
                3,
            ),
            dtype=np.int64,
        )
        self._actions: list[int] = [1, 0, -1]

    def report_exploration_counts(self, state: list[int]) -> list[int]:
        """
        Check to see how many times each action has been explored in this state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        explored_count (array of 3 ints):
          number of times that each action has been explored from this state.
          The mapping from actions to integers is up to you, but there must be three of them.
        """
        explored_count = (
            self.N[state[0], state[1], state[2], state[3], state[4]].flatten().tolist()
        )

        return explored_count

    def choose_unexplored_action(self, state: list[int]) -> int | None:
        """
        Choose an action that has been explored less than nfirst times.
        If many actions are underexplored, you should choose uniformly
        from among those actions; don't just choose the first one all
        the time.

        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
           These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar): either -1, or 0, or 1, or None
          If all actions have been explored at least n_explore times, return None.
          Otherwise, choose one uniformly at random from those w/count less than n_explore.
          When you choose an action, you should increment its count in your counter table.
        """
        explored_count: list[int] = self.report_exploration_counts(state)

        if min(explored_count) >= self._nfirst:
            return None
        else:
            under_explored: list[int] = [
                i for i, x in enumerate(explored_count) if x < self._nfirst
            ]
            action: int = random.choice(under_explored)
            self.N[state[0], state[1], state[2], state[3], state[4], action] += 1
            return action

    def report_q(self, state: list[int]) -> ndarray:
        """
        Report the current Q values for the given state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        Q (array of 3 floats):
          reward plus expected future utility of each of the three actions.
          The mapping from actions to integers is up to you, but there must be three of them.
        """
        return self.Q[state[0], state[1], state[2], state[3], state[4]]

    def q_local(self, reward: float, newstate: list[int]):
        """
        The update to Q estimated from a single step of game play:
        reward plus gamma times the max of Q[newstate, ...].

        @param:
        reward (scalar float): the reward achieved from the current step of game play.
        newstate (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        Q_local (scalar float): the local value of Q
        """
        return reward + self._gamma * max(self.report_q(newstate))

    def learn(self, state: list[int], action: int, reward: float, newstate: list[int]):
        """
        Update the internal Q-table on the basis of an observed
        state, action, reward, newstate sequence.

        @params:
        state: a list of 5 numbers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle.
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 numbers, in the same format as state

        @return:
        None
        """
        self.Q[state[0], state[1], state[2], state[3], state[4], action] = self.Q[
            state[0], state[1], state[2], state[3], state[4], action
        ] + self._alpha * (
            self.q_local(reward, newstate)
            - self.Q[state[0], state[1], state[2], state[3], state[4], action]
        )

    def save(self, filename: str):
        """
        Save your Q and N tables to a file.
        This can save in any format you like, as long as your "load"
        function uses the same file format.  We recommend numpy.savez,
        but you can use something else if you prefer.

        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        """
        with open(filename, "wb") as f:
            np.savez(f, Q=self.Q, N=self.N)

    def load(self, filename: str):
        """
        Load the Q and N tables from a file.
        This should load from whatever file format your save function
        used.  We recommend numpy.load, but you can use something
        else if you prefer.

        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        """
        with open(filename, "rb") as f:
            data = np.load(f)
            self.Q = data["Q"]
            self.N = data["N"]

    def exploit(self, state: list[int]) -> tuple[int, float]:
        """
        Return the action that has the highest Q-value for the current state, and its Q-value.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar int): either -1, or 0, or 1.
          The action that has the highest Q-value.  Ties can be broken any way you want.
        Q (scalar float):
          The Q-value of the selected action
        """
        return self._actions[np.argmax(self.report_q(state))], np.max(
            self.report_q(state)
        )

    def act(self, state):
        """
        Decide what action to take in the current state.
        If any action has been taken less than nfirst times, then choose one of those
        actions, uniformly at random.
        Otherwise, with probability epsilon, choose an action uniformly at random.
        Otherwise, choose the action with the best Q(state,action).

        @params:
        state: a list of 5 integers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        """
        raise RuntimeError("You need to write this!")


class deep_q:
    def __init__(self, alpha, epsilon, gamma, nfirst):
        """
        Create a new deep_q learner.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a deep learning model that will accept
        (state,action) as input, and estimate Q as the output.

        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting

        @return:
        None
        """
        raise RuntimeError("You need to write this!")

    def act(self, state):
        """
        Decide what action to take in the current state.
        You are free to determine your own exploration/exploitation policy --
        you don't need to use the epsilon and nfirst provided to you.

        @params:
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        """
        raise RuntimeError("You need to write this!")

    def learn(self, state, action, reward, newstate):
        """
        Perform one iteration of training on a deep-Q model.

        @params:
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 floats, in the same format as state

        @return:
        None
        """
        raise RuntimeError("You need to write this!")

    def save(self, filename):
        """
        Save your trained deep-Q model to a file.
        This can save in any format you like, as long as your "load"
        function uses the same file format.

        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        """
        raise RuntimeError("You need to write this!")

    def load(self, filename):
        """
        Load your deep-Q model from a file.
        This should load from whatever file format your save function
        used.

        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        """
        raise RuntimeError("You need to write this!")
