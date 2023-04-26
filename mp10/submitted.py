"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
"""
from itertools import product
from typing import Final, TypeAlias

import numpy as np
from numpy import ndarray

from utils import GridWorld

Location: TypeAlias = tuple[int, int]
Move: TypeAlias = tuple[int, int]

epsilon = 1e-3
moves: list[Move] = [(0, -1), (-1, 0), (0, 1), (1, 0)]


def compute_transition_matrix(model: GridWorld):
    """
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    """
    M: Final[int] = model.M
    N: Final[int] = model.N

    P = np.zeros((M, N, 4, M, N))
    loc_combinations: list[Location] = list(product(range(M), range(N)))

    for r, c in loc_combinations:
        if model.T[r, c]:
            continue
        else:
            for a, correct_move in enumerate(moves):
                wrong_move_left = moves[(a - 1) % 4]
                wrong_move_right = moves[(a + 1) % 4]

                possible_moves = [correct_move, wrong_move_left, wrong_move_right]

                for i, possible_move in enumerate(possible_moves):
                    if (
                        r + possible_move[0] < 0
                        or r + possible_move[0] >= M
                        or c + possible_move[1] < 0
                        or c + possible_move[1] >= N
                        or model.W[r + possible_move[0], c + possible_move[1]]
                    ):
                        P[r, c, a, r, c] += model.D[r, c, i]
                    else:
                        P[
                            r, c, a, r + possible_move[0], c + possible_move[1]
                        ] += model.D[r, c, i]

    return P


def update_utility(model: GridWorld, P: ndarray, U_current: ndarray):
    """
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    """
    gamma: ndarray = model.gamma
    loc_combinations: list[Location] = list(product(range(model.M), range(model.N)))

    U_next = np.zeros((model.M, model.N))

    for s in loc_combinations:
        util_sums: list[float] = []
        for a, _ in enumerate(moves):
            # S_next: list[Location] = [
            #     (s[0] + move[0], s[1] + move[1])
            #     for move in moves
            #     if s[0] + move[0] >= 0
            #     and s[0] + move[0] < model.M
            #     and s[1] + move[1] >= 0
            #     and s[1] + move[1] < model.N
            # ]
            # print(s, S_next)
            util_sum = np.sum(
                [
                    P[s[0], s[1], a, s_next[0], s_next[1]]
                    * U_current[s_next[0], s_next[1]]
                    for s_next in loc_combinations
                ]
            )
            util_sums.append(util_sum)

        U_next[s[0], s[1]] += model.R[s[0], s[1]] + gamma * np.max(util_sums)
    # print(U_next)
    return U_next


def value_iteration(model: GridWorld):
    """
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    """
    P: ndarray = compute_transition_matrix(model)
    U: ndarray = np.zeros((model.M, model.N))

    while True:
        U_current = update_utility(model, P, U)
        if np.max(np.abs(U - U_current)) < epsilon:
            break
        U = U_current

    return U


if __name__ == "__main__":
    import utils

    model = utils.load_MDP("models/small.json")
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
