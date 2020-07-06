import numpy as np
from Game import Game
import time

# constants for board dims
num_rows = 6
num_cols = 7

# time (seconds) of search per cpu turn
search_time = 15


def main():

    # initialize board -> node -> root of tree
    root_board = np.zeros((num_rows, num_cols), dtype=int)
    node = Node(root_board)
    tree = Tree(node)

    # begin tree search
    print("Lets play Connect Four!\nOne moment to initialize the game please.")
    tree.run_tree_builder(30, tree.root)
    tree.game_depth = 0

    # create game instance
    game = Game(num_rows, num_cols)
    game.board = np.copy(node.board)

    # main loop
    while True:

        # function to run tree search, and to take action based on search
        # node replaced to update current location in game-tree
        print("\nCPU is thinking...")
        tree.run_tree_builder(search_time, node)
        # node.get_children_weights()
        game.move(node.get_max_child(), 1)
        game.display_board()
        node = node.children[node.find_child(game.board)]

        # check if game over
        if game.check_game_over()[0]:
            break

        # tree search continues while user evaluates potential moves
        # user inputs move and node in game-tree is updated
        tree.run_tree_builder(2, node)
        move = int(input("column:"))
        game.move(move, 2)
        game.display_board()
        node = node.children[node.find_child(game.board)]

        # check if game over
        if game.check_game_over()[0]:
            break


class Tree:

    def __init__(self, node):
        self.root = node
        self.game_depth = 0
        self.start_size = 0
        self.current_size = 0
        self.size_difference_constant = 280

    # loops through tree search for specified duration
    # able to start search from any node (as input)
    def run_tree_builder(self, duration, node):
        start_time = time.time()
        self.start_size = np.copy(self.current_size)
        while time.time() - start_time < duration:
            self.monte_carlo_tree_search(node)
            self.current_size += 1
        self.game_depth += 1

    # performing the four steps of MCTS
    def monte_carlo_tree_search(self, node):
        local_root = node
        node, board, count, game_over_tuple = self.selection(node, local_root)
        if not game_over_tuple[0]:
            node = self.expansion(node, board)
            game_over_tuple = self.simulation(node, count)
        self.backpropagation(local_root, node, game_over_tuple)

    # let us explore states and exploit what we learn
    def selection(self, node, local_root):
        temp_game = Game(num_rows, num_cols)
        temp_game.board = np.copy(node.board)
        count = self.game_depth

        # check for end of game state
        game_over_tuple = temp_game.check_game_over()
        if game_over_tuple[0]:
            return node, temp_game.board, count, game_over_tuple

        # loop until we find bottom of tree
        while True:
            found = False
            # TODO check this logic vs agent performance
            if self.current_size - self.start_size >= self.size_difference_constant or node != local_root:
                # upper confidence bound selection
                column = self.get_selection(node, count)
            else:
                column = (self.current_size - self.start_size) % num_cols
                while temp_game.is_column_full(column):
                    column = np.random.randint(num_cols)
            temp_game.board = temp_game.move(column, count % 2 + 1)

            # search for new node based on new state
            for i in range(len(node.children)):
                if np.array_equal(node.children[i].board, temp_game.board):
                    node = node.children[i]
                    found = True
                    count += 1
                    break

            # if node does not exist -> found bottom of tree
            if not found:
                break

        return node, temp_game.board, count, game_over_tuple

    # create new node from given game state
    @staticmethod
    def expansion(node, board):
        child_node = Node(board)
        child_node.parent = node
        node.children.append(child_node)
        return child_node

    # simulate through unknown environment
    @staticmethod
    def simulation(node, count):
        game = Game(num_rows, num_cols)
        game.board = np.copy(node.board)

        # take random actions until game_over
        while True:
            game_over_tuple = game.check_game_over()
            if game_over_tuple[0]:
                break
            # todo make this a function
            while True:
                column = np.random.randint(num_cols)
                if not game.is_column_full(column):
                    break
            game.board = game.move(column, count % 2 + 1)
            count += 1

        return game_over_tuple

    # use game_over results as feedback for agent
    def backpropagation(self, local_root, node, game_over_tuple):
        if game_over_tuple[1] == 1:
            # win logic
            self.win_backpropagation_helper(local_root, node)
        else:
            # loss logic
            self.loss_backpropagation_helper(local_root, node)

    def win_backpropagation_helper(self, local_root, node):
        # base case
        if node == local_root:
            node.add_win()
            return local_root

        # recursive call to parent node
        node.add_win()
        node.parent = self.win_backpropagation_helper(local_root, node.parent)
        return node

    def loss_backpropagation_helper(self, local_root, node):
        # base case
        if node == local_root:
            node.add_loss()
            return local_root

        # recursive call to parent node
        node.add_loss()
        node.parent = self.loss_backpropagation_helper(local_root, node.parent)
        return node

    # upper confidence bound selection
    def get_selection(self, node, count):
        scores_size = len(node.children)
        if not scores_size:
            return np.random.randint(num_cols)

        # player 1
        if count % 2 + 1 == 1:
            return self.explore_exploit_minimax(node, scores_size, True)

        # player 2
        else:
            return self.explore_exploit_minimax(node, scores_size, False)

    # trade-off management between exploration and exploitation
    @staticmethod
    def explore_exploit_minimax(node, scores_size, maximize):
        child_node_scores = np.zeros(scores_size)

        # UCB equation for MCTS for all child nodes
        for i in range(scores_size):
            explore = 0.75 * np.sqrt((2 * np.log(node.total)) / node.children[i].total)
            if maximize:
                exploit = node.children[i].wins / node.children[i].total
            else:
                exploit = 1 - (node.children[i].wins / node.children[i].total)
            child_node_scores[i] = explore + exploit

        # maximize UCB for player 1
        if maximize:
            return node.find_child(node.children[np.argmax(child_node_scores)].board)

        # minimize UCB for player 2
        else:
            return node.find_child(node.children[np.argmin(child_node_scores)].board)


class Node:

    def __init__(self, board):
        self.board = board
        self.wins = 0
        self.total = 0
        self.children = []
        self.parent = None

    # helper function for win
    def add_win(self):
        self.wins += 1
        self.total += 1

    # helper function for loss
    def add_loss(self):
        self.total += 1

    # search children for given state
    def find_child(self, board):
        index = None

        for i in range(len(self.children)):
            if np.array_equal(self.children[i].board, board):
                index = i

        return index

    # return child with maximum win probability
    def get_max_child(self):
        maximum = 0
        index = -1

        for i in range(len(self.children)):
            child = self.children[i]
            if child.wins / child.total >= maximum:
                maximum = child.wins / child.total
                index = i

        for i in range(num_cols):
            game1 = Game(num_rows, num_cols)
            game2 = Game(num_rows, num_cols)

            game1.board = np.copy(self.board)
            game2.board = np.copy(self.board)

            game1.move(i, 1)
            game2.move(i, 2)
            if game1.check_game_over()[0] or game2.check_game_over()[0]:
                return i

        return index

    # analytical function to help debug
    def get_children_weights(self):
        print("\nChild Weights")
        for i in range(len(self.children)):
            child = self.children[i]
            print(i, "Wins:", child.wins, "total:", child.total, "win_prob:", child.wins / child.total)


main()
