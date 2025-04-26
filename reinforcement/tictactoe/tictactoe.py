import numpy as np

class Board:
    def __init__(self, size=3, time=3, num_to_win=3):
        self.board = self._init_board(size=size, time=time)
        self.size = size
        self.time = time
        self.num_to_win = num_to_win

    def __repr__(self):

        # convert board to 2d representation
        board = self.get_board_2d()

        to_print = ""
        n, m = board.shape[0], board.shape[1]
        for i in range(n):
            for j in range(m):
                val = board[i, j]
                if j==0:
                    to_print += " "
                if val == 1:
                    to_print += "X "
                elif val == -1:
                    to_print += "O "
                else:
                    to_print += "  "
                
                if j!=m-1:
                    to_print += "| "
                    
            to_print += "\n"
            if i!=n-1:
                to_print += "---+"*(m-1) + "---\n" 
            
        return to_print
    
    def get_board_2d(self):
        # collapse board from 3D to 2D
        def first_nonzero(v):
            for i in v:
                if not i==0:
                    return i
            return 0
        board = np.apply_along_axis(first_nonzero, axis=2, arr=self.board)
        return board

    def _init_board(self, size, time):
        """
            Create 3D board
            dim1/dim2 are regular board (size x size)
            dim3 represents time that move was made

            Args:
                size: size of one side of board (board will be side x side)
                time: amount of time before each move disappears
        """
        return np.zeros((size,size,time*2))


    def make_move(self, move, player):
        """
            Update board with move by player

            Args:
                board: 3x3 np array of current board
                move:  index1,index2 of where on the board to move
        """

        if not self.check_valid_move(move):
            print("\nNot a valid move!\n")
            return False
        
        # insert empty board at time=0
        empty_board = np.zeros((self.size, self.size, 1))
        new_board = np.concatenate([empty_board, self.board], axis=2)

        # shave off board slice of oldest moves
        oldest_index = new_board.shape[2]-1
        new_board = np.delete(new_board, oldest_index, axis=2)

        # enter new move
        new_board[move[0], move[1], 0] = player

        self.board = new_board
        return True
    
    def check_bounds(self, x, y):

        board_shape = self.board.shape
        n, m = board_shape[0], board_shape[1]

        return 0 <= x < n and 0 <= y < m

    def check_valid_move(self, move):

        # check out of range
        if not self.check_bounds(move[0], move[1]):
            return False

        # get the 2D version of the board
        board = self.get_board_2d()

        # return if space being moved is empty
        return board[move[0], move[1]]==0
    
    def check_win(self):

        # convert board to 2D representation
        board_2d = self.get_board_2d()
        n, m = board_2d.shape[0], board_2d.shape[1]

        # define number in a row needed to win
        num_to_win = self.num_to_win

        # dfs search
        def dfs_win_search(x, y, dx, dy, player, count):
            """ 
                Recursively check for win, starting from point (dx, dy)
                and moving in the direction (dx, dy)
            """

            # stopping condition for win found
            if count==num_to_win:
                return player
            
            # get new point
            nx, ny = x + dx, y+dy

            # if in bounds and move by same player, check further
            if self.check_bounds(nx, ny) and board_2d[nx, ny] == player:
                return dfs_win_search(nx, ny, dx, dy, player, count + 1)
            
            return 0

        # dfs search from each point
        #  (only need to search half of directions)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for i in range(n):
            for j in range(m):
                player = board_2d[i, j]

                # ignore empty starting point
                if player==0: continue

                for (dx, dy) in directions:
                    if dfs_win_search(i, j, dx, dy, player, 1):
                        return player
        return 0