import math
from typing import Callable, Literal, NamedTuple
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion

Grid = Literal[1, 2, 3, 4, 5, 6, 7, 8]
Location = tuple[Grid, Grid]


class Piece(NamedTuple):
    x: Grid
    y: Grid
    type: Literal["p", "r", "n", "b", "q", "k"]


Board = tuple[list[Piece], list[Piece]]
Flag = list[bool]
Flags = tuple[Flag, Flag | None]

Promote = Literal[None, "q"]
Move = tuple[Location, Location, Promote]


def generateMoves(side: bool, board: Board, flags: Flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(
                None, side, board, fro, to, single=True)
            yield [fro, to, promote]


###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.


def random(side, board, flags, chooser):
    """
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    """
    moves = [move for move in generateMoves(side, board, flags)]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(
            side, board, move[0], move[1], flags, move[2]
        )
        value = evaluate(newboard)
        return (value, [move], {encode(*move): {}})
    else:
        return (evaluate(board), [], {})


###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.

class Node:
    def __init__(self, value: float | None = None, children: list['Node'] | None = None, move_path: list[str] | None = None):
        self.value: float | None = value
        self.children: list['Node'] = children or []
        self.move_path: list[str] = move_path or []


Root = Literal['min', 'max']


def find_optimal_value(node: Node, depth: int, is_maximizing_player: bool) -> tuple[float, list[str]]:
    if depth == 2:
        pass
    best_value: float = node.value
    best_path: float = node.move_path
    if depth == 0 or len(node.children) == 0:
        pass
    else:
        if is_maximizing_player:
            best_value = -math.inf
            for child in node.children:
                child_value, move_path = find_optimal_value(
                    child, depth - 1, False)
                if child_value > best_value:
                    best_path = move_path
                else:
                    pass
                best_value = max(
                    best_value, child_value) if child_value else best_value

        else:
            best_value = math.inf
            for child in node.children:
                child_value, move_path = find_optimal_value(
                    child, depth - 1, True)

                if child_value < best_value:
                    best_path = move_path
                else:
                    pass
                best_value = min(
                    best_value, child_value) if child_value else best_value

    return best_value, best_path


def minimax(side: bool, board: Board, flags: Flags, depth: int) -> tuple[float, list[str], dict[str, dict]]:
    """
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    """
    is_maximizing_player: bool = not side
    nodes: list[Node] = [[Node()]]
    moveTree: dict[str, dict]
    for i in range(depth):
        depth_nodes: list[Node] = []
        current_depth_nodes: list[Node] = nodes[i]

        for node in current_depth_nodes:
            newside: bool = side
            newboard: Board = board
            newflags: Flags = flags

            if node.move_path:
                for move in node.move_path:
                    decoded_move: Move = decode(move)
                    newside, newboard, newflags = makeMove(
                        newside, newboard, decoded_move[0], decoded_move[1], newflags, decoded_move[2])

            else:
                pass

            new_moves: list[Move] = generateMoves(newside, newboard, newflags)

            if new_moves:

                new_encoded_moves: list[str] = [
                    encode(*move) for move in new_moves]

                if i == 0:
                    moveTree = {move: {} for move in new_encoded_moves}
                else:
                    move_dict: dict[str, dict] = {move: {}
                                                  for move in new_encoded_moves}
                    result = moveTree
                    for key in node.move_path:
                        result = result[key]

                    result.update(move_dict)

                children: list[Node]
                if i != depth - 1:
                    children = [
                        Node(None, None, node.move_path + [encode(*move)]) for move in generateMoves(newside, newboard, newflags)]

                else:
                    children = []
                    for move in generateMoves(newside, newboard, newflags):
                        final_board: Board
                        _, final_board, _ = makeMove(
                            newside, newboard, move[0], move[1], newflags, move[2])
                        value: float = evaluate(final_board)
                        children.append(
                            Node(value, None, node.move_path + [encode(*move)]))

                node.children = children
                depth_nodes += children
                nodes.append(depth_nodes)

            else:
                value: float = evaluate(newboard)
                node.value = value

    best_value, best_path = find_optimal_value(
        nodes[0][0], depth, is_maximizing_player)

    moveList = [decode(move) for move in best_path]
    if moveList == [[[7, 1], [6, 3], None]]:
        pass
    return best_value, moveList, moveTree


def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    """
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    """
    raise NotImplementedError("you need to write this!")


def stochastic(side, board, flags, depth, breadth, chooser):
    """
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths
      chooser: a function similar to random.choice, but during autograding, might not be random.
    """
    raise NotImplementedError("you need to write this!")
