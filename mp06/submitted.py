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

class Node(NamedTuple):
    def __init__(self, value: int | None = None, children: list['Node'] | None = None):
        self.value: int | None = value
        self.children: list['Node'] = children or []


Root = Literal['min', 'max']


def find_optimal_value(node: Node, depth: int, is_maximizing_player: bool) -> int:
    if depth == 0 or len(node.children) == 0:
        return node.value

    if is_maximizing_player:
        best_value = -math.inf
        for child in node.children:
            child_value = minimax(child, depth - 1, False)
            best_value = max(
                best_value, child_value) if child_value else best_value
        return best_value
    else:
        best_value = math.inf
        for child in node.children:
            child_value = minimax(child, depth - 1, True)
            best_value = min(
                best_value, child_value) if child_value else best_value
        return best_value


def minimax(side: bool, board: Board, flags: Flags, depth: int):
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
    root: Root = 'min' if side else 'max'
    white_pieces: list[Piece]
    black_pieces: list[Piece]
    white_pieces, black_pieces = board
    nodes: list[list[Node]] = [[Node(None, None, None)]]
    moveTree: dict[str, dict]
    for i in range(depth):
        depth_nodes: list[Node]
        depth_nodes = []
        current_depth_nodes: list[Node] = nodes[i]

        for node in current_depth_nodes:
            newside: bool = side
            newboard: Board = board
            newflags: Flags = flags

            if node.move != None:
                encoded_move_path: list[str] = []
                tracked_node = node

                while tracked_node is not None:
                    encoded_move_path.append(tracked_node.move)
                    tracked_node = tracked_node.parent

                encoded_move_path.reverse()

                for move in encoded_move_path:
                    decoded_move: Move = decode(move)
                    newside, newboard, newflags = makeMove(
                        newside, newboard, decoded_move[0], decoded_move[1], newflags, decoded_move[2])

            else:
                pass

            new_moves: list[Move] = generateMoves(newside, newboard, newflags)
            new_encoded_moves: list[str] = [
                encode(*move) for move in new_moves]

            if i == 0:
                moveTree = {move: {} for move in new_encoded_moves}
            else:
                move_dict: dict[str, dict] = {move: {}
                                              for move in new_encoded_moves}
                result = moveTree
                for key in encoded_move_path:
                    result = result[key]

                result.update(move_dict)

            if i != depth - 1:
                children: list[Node] = [
                    Node(None, encode(*move), node) for move in generateMoves(newside, newboard, newflags)]
                depth_nodes += children
            else:
                children: list[Node] = []
                values: list[int] = []
                for move in generateMoves(newside, newboard, newflags):
                    final_board: Board
                    _, final_board, _ = makeMove(
                        newside, newboard, move[0], move[1], newflags, move[2])
                    value = evaluate(board)
                    values.append(value)
                    children.append(Node(value, encode(move), node))

                optimal_value = min_max_depth(root, depth-1)(values)
                nodes[i-1].remove(node)
                nodes[i-1].append(Node(optimal_value,
                                  node.move, node.parent))

        nodes.append(depth_nodes)


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
