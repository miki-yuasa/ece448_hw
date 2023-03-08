import math

# Define the tree structure


class Node:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children or []

# Define the Minimax algorithm


def minimax(node, depth, is_maximizing_player):
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


# Define the tree
root = Node()
root.value = None

node_a = Node()
node_a.value = None
root.children.append(node_a)

node_b = Node()
node_b.value = None
root.children.append(node_b)

node_c = Node()
node_c.value = None
node_a.children.append(node_c)

node_d = Node()
node_d.value = None
node_a.children.append(node_d)

node_e = Node()
node_e.value = 3
node_b.children.append(node_e)

node_f = Node()
node_f.value = None
node_c.children.append(node_f)

node_g = Node()
node_g.value = 6
node_c.children.append(node_g)

node_h = Node()
node_h.value = 2
node_d.children.append(node_h)

node_i = Node()
node_i.value = None
node_d.children.append(node_i)

node_j = Node()
node_j.value = 9
node_f.children.append(node_j)

node_k = Node()
node_k.value = 1
node_i.children.append(node_k)

node_l = Node()
node_l.value = 8
node_i.children.append(node_l)

# Call the Minimax algorithm with a depth of 3 and starting from the root node
result = minimax(root, 3, True)
print(result)  # Prints the maximum value, which is 6 in this case
