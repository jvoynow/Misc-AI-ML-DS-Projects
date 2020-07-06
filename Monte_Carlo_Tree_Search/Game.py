import numpy as np


class Game:
    def __init__(self, num_rows, num_cols):
        self.board = np.zeros((num_rows, num_cols), dtype=int)
        self.pieces = ['_', 'X', 'O']
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.win_length = 4

    # make move on board given column input
    def move(self, x, player):
        i = 0
        while i < self.num_rows:
            if self.board[i][x] != 0:
                break
            i += 1
        # either existing move found or row empty
        # insert move above current i
        self.board[i-1][x] = player
        return self.board

    # check if move is valid
    # functionality to check for for full column still needed

    def valid_move(self, x):
        return x and x < self.num_cols and not self.is_column_full(x)

    # check column for open space to drop a piece
    def is_column_full(self, x):
        return self.board[0][x] == [1 or 2]

    def check_game_over(self):

        # check vertical wins for either player
        game_over, winner = self.vertical_horizontal_check(self.board)
        if game_over:
            return game_over, winner

        # check horizontal wins for either player
        game_over, winner = self.vertical_horizontal_check(self.board.T)
        if game_over:
            return game_over, winner

        # diagonal win in one direction, for both players
        game_over, winner = self.diagonal_check(self.board)
        if game_over:
            return game_over, winner

        # diagonal win in other direction, for both players
        game_over, winner = self.diagonal_check(np.fliplr(self.board))
        if game_over:
            return game_over, winner

        # check for full board -> tie
        if np.sum(self.board == 0) == 0:
            return True, 0

        return game_over, winner

    def vertical_horizontal_check(self, temp_board):
        for row in temp_board:
            for i in range(self.num_cols - 3):
                for j in range(1, 3):
                    if np.sum(row[i:i + self.win_length] == j) == self.win_length:
                        return True, j
            for i in range(self.num_cols - 3):
                for j in range(1, 3):
                    if np.sum(row[i:i + self.win_length] == j) == self.win_length:
                        return True, j

        return False, 0

    def diagonal_check(self, temp_board):
        for i in range(-2, 4):
            diagonal = np.diagonal(temp_board, i)
            for j in range(len(diagonal) - 3):
                for k in range(1, 3):
                    if np.sum(diagonal[j:j + self.win_length] == k) == self.win_length:
                        return True, k

        return False, 0

    # uses pieces list to print a more readable board
    def display_board(self):
        for i in self.board:
            string = ""
            for j in i:
                j = self.pieces[j]
                string += str(j) + "  "
            print(string)
        print("-" * 19)
