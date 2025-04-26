from tictactoe import Board


board = Board(size=3, time=3)
player = 1
while True:

    print(board)
    
    
    move1, move2 = map(int, input("Enter two move indices (separated by space): ").split())

    
    success = board.make_move([move1, move2], player)
    if not success:
        continue

    player_win = board.check_win()
    if player_win != 0:
        player = "X" if player_win==1 else "O"
        print(board)
        print(f"\nPlayer {player} wins!\n")
        break

    player = -1 if player==1 else 1
