import numpy as np
import time
import random

def main():
    # --- 1. Load the constants
    board_size, win_length, EMPTY, BLACK, WHITE = constants()

    # --- 2. Initialize the board with BLACK already played in the center
    board, center = initial_board(board_size)
    
    # --- 3. Display welcome message and initial board
    display_welcome_message()
    print_board(board, board_size)
    print()
    
    # --- 4. Constants
    player_symbol,ai_symbol,player_turn=choose_side(BLACK,WHITE)
    current_player = WHITE  # Next turn belongs to White
    black_second_move=True
    restricted_zone = (center - 3, center + 3)
    last_move=(7,7)
    moves=2
    
    # --- 5. Game: 
    while True:
        if player_turn:
            # --- Human move
            print(f'Move: {moves}')
            print("Your turn!")
            move = input("Enter your move (e.g., H7): ").strip().upper()
            try:
                row, col = ord(move[0]) - 65, int(move[1:])
                if is_valid_move(board, row, col, board_size, EMPTY): #Make sure the move is valid
                    if player_symbol==BLACK and black_second_move and ( #if Human is black and its the second move, restrict the playing zone
                            restricted_zone[0] <= row <= restricted_zone[1] and
                            restricted_zone[0] <= col <= restricted_zone[1]):
                        print("Invalid move. Black's second move cannot be within the 7x7 restricted zone. Try again.")
                        continue
                    else:
                        board[row][col] = current_player
                else:
                    print("Invalid move. Try again.")
                    continue
            except:
                print("Invalid format. Try again.")
                continue
            if player_symbol==BLACK:
                black_second_move=False
        else:
            # --- AI move
            print(f'Move: {moves}')
            print("AI's turn!")
            if ai_symbol==WHITE:
                #MODIF Ici/
                start_time = time.time()
                if moves==2:
                    row,col=12,12
                else:
                    row, col = find_best_move(
                        board=board,
                        current_player=current_player,
                        board_size=board_size,
                    )
                end_time = time.time()
            else:
                if black_second_move==True: #if its the second move, we restrict the zone and play a predefined move 
                    start_time = time.time()
                    #check if the move played by white is in the 7x7 zone if it is our algorithm is useless and better play a predefined move
                    if (restricted_zone[0] <= last_move[0] <= restricted_zone[1] and restricted_zone[0] <= last_move[1] <= restricted_zone[1]):
                        row, col = 11,7
                    else: #if out of zone we can apply our algorithm 
                        row,col=find_best_move(board, current_player, board_size, black_second_move)
                        end_time = time.time()
                    end_time = time.time()
                    black_second_move=False
                else:
                    start_time = time.time()
                    row, col = find_best_move(
                        board=board,
                        current_player=current_player,
                        board_size=board_size,
                    )
                    end_time = time.time()
                    
            if row != -1 and col != -1:
                board[row][col] = current_player
                elapsed_time = end_time - start_time
                print(f"The AI took {elapsed_time:.2f} seconds")
                print(f"AI played at {chr(65 + row)}{col}.")

        # --- Print board and check for winner or draw
        print_board(board, board_size)
        winner = has_winner(board, board_size, EMPTY, win_length)
        if winner==ai_symbol:
            the_winner='AI wins'
        else:
            the_winner='You win'
        if winner:
            print(f"{the_winner}!")
            break
        if board_is_full(board, EMPTY) or moves>= 120:
            print("It's a draw!")
            break

        # Swap turns
        player_turn = not player_turn
        current_player = WHITE if current_player == BLACK else BLACK
        last_move = (row,col)
        moves+=1


#%%Game and initialisation functions 

def constants():
    board_size = 15
    win_length = 5
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    return board_size, win_length, EMPTY, BLACK, WHITE

def initial_board(board_size):
    """
    Create a board_size x board_size array, all zeros, then place a BLACK piece at center.
    """
    board = create_board(board_size, 0)
    center = board_size // 2
    board[center][center] = 1
    return board, center

def create_board(board_size, empty):
    return np.full((board_size, board_size), empty)

def display_welcome_message():
    print("Welcome to Gomoku!")
    print("Black starts at the center of the board.")
    print("White plays next. First to 5 in a row wins!")

def print_board(board, board_size):
    """
    Print the board in a more readable format, with row labels (A, B, C, ...) and column labels (0..14).
    """
    # Print the top header row of column indices
    headers = "   " + " ".join(f"{i:2}" for i in range(board_size))
    print(headers)
    # Print each row
    for i, row in enumerate(board):
        row_label = chr(65 + i)
        row_str = " ".join(
            "X " if cell == 1 else ("O " if cell == 2 else ". ")
            for cell in row
        )
        # Add row label to left
        print(f"{row_label:2} {row_str}")
     
def choose_side(BLACK,WHITE):
    """
    Allows player to choose his or her side 
    """
    while True:
        player_choice = input("Do you want to play as Black (X) or White (O)? Enter X or O: ").strip().upper()
        if player_choice in ['X', 'O']:
            break
        print("Invalid choice. Please enter X or O.")
    player_symbol = BLACK if player_choice == 'X' else WHITE
    ai_symbol = WHITE if player_symbol == BLACK else BLACK
    
    if player_symbol==BLACK:
        player_turn=False
    else:
        player_turn=True
    return player_symbol,ai_symbol,player_turn

def is_valid_move(board, row, col, board_size, empty):
    """
    Checks that the chosen row,col is inside the board and is empty.
    """
    if 0 <= row < board_size and 0 <= col < board_size:
        return board[row][col] == empty
    return False

#%%AI functions 

def find_best_move(board, current_player, board_size, second_move=False):
    """
    Finds the best move for the 'current_player' using minimax with alpha-beta.
    """
    #Constants
    center = board_size // 2
    restricted_zone = (center - 3, center + 3)
    depth = 3
    best_score = float("-inf")
    alpha, beta = float("-inf"), float("inf")
    best_move_row, best_move_col = -1, -1
    max_depth = 6
    start_time=time.time()
    timeout=5
    
    if second_move: #if its black's second move and white has played oustide 7x7, we filter the possible moves 
        candidate_cols, candidate_rows = get_coords_around(board_size, board)
        filtered_moves = []
        for (xc, yc) in zip(candidate_cols, candidate_rows):
            if not (restricted_zone[0] <= yc <= restricted_zone[1] and restricted_zone[0] <= xc <= restricted_zone[1]):
                filtered_moves.append((xc, yc))
        candidate_cols, candidate_rows = zip(*filtered_moves) if filtered_moves else ([], [])
    else: 
        timer1=time.time()
        # Gather candidate moves
        candidate_cols, candidate_rows = get_coords_around(board_size, board,2)
        coords_around = set(zip(candidate_rows, candidate_cols))

        opponent = 2 if current_player == 1 else 1
        # Prevent or play immediate threats: 
        move = check_for_immediate_win(board, current_player, coords_around)
        if move is not None:
            print('win')
            return move[0], move[1]
        
        move_to_block = check_for_immediate_win(board, opponent, coords_around)
        if move_to_block is not None:
            print('block a win')
            return move_to_block[0], move_to_block[1]
        
        move = check_for_immediate_win2(board, current_player, coords_around)
        if move is not None:
            print('create a color color color color empty')
            return move[0], move[1]
        
        move_to_block=check_for_immediate_win2(board, opponent, coords_around)
        if move_to_block is not None:
            print('block a color color color color empty')
            return move_to_block[0], move_to_block[1]
        
        move = check_for_immediate_win3(board, current_player, coords_around)
        if move is not None:
            return move[0], move[1]
        
        move_to_block=check_for_immediate_win3(board, opponent, coords_around)
        if move_to_block is not None:
            return move_to_block[0], move_to_block[1]
        timer2=time.time()
        print(f'time spent on initial checks: {timer2-timer1}')
        
        candidate_cols, candidate_rows = get_coords_around(board_size, board)
        
    # --- 1. Compute a local_heuristic_score for ordering so that alpha beta more effective
    move_candidates = []
    for (cx, cy) in zip(candidate_cols, candidate_rows):
        if board[cy][cx] == 0:
            local_score = quick_heuristic(board, cx, cy, current_player)
            move_candidates.append((cx, cy, local_score))

    # --- 2. Sort moves in DESCENDING order by local_score
    move_candidates.sort(key=lambda m: m[2], reverse=True)

    # --- 3. Explore the moves
    best_moves_by_depth = {}
    #Iterative depth search
    for depth in range(3, max_depth + 1):
        alpha, beta = float("-inf"), float("inf")
        best_score = float("-inf")
        #Find the best move at a certain depth
        for (cx, cy, _) in move_candidates:
            if time.time() - start_time > timeout: #time restriction
                if best_moves_by_depth: #return the best move calculated by the last depth if possible 
                    best_final_move = max(best_moves_by_depth.values(), key=lambda x: x[3])
                    print(f'we play {best_final_move}')
                    return best_final_move[0], best_final_move[1]
                else:
                    return best_move_row, best_move_col
            board[cy][cx] = current_player
            score = minimax(board, depth - 1, alpha, beta, False, current_player, start_time, timeout)
            board[cy][cx] = 0
            if score > best_score:
                best_score = score
                best_move_row, best_move_col = cy, cx
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        print(f'depth = {depth} and depth move is: {best_move_row},{best_move_col} ')
        best_moves_by_depth[depth] = (best_move_row, best_move_col, best_score, depth)
    best_final_move = max(best_moves_by_depth.values(), key=lambda x: x[2])
    print(f'we play {best_final_move}')
    
    return best_final_move[0], best_final_move[1]


def quick_heuristic(board, x, y, player):
    """
    A simple heuristic for move-ordering.
    """
    count_same_color = 0
    size = len(board)
    radius = 2 # Check cells in a 2-cell radius 
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < size and 0 <= nx < size:
                if board[ny][nx] == player:
                    count_same_color += 1
    return count_same_color

def check_for_immediate_win(board, player, coords_around):
    """
    Returns (row, col) if the 'player' can immediately win on this turn. Otherwise, returns None.
    """
    board_size = len(board)
    empty = 0
    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] == empty and (r,c) in coords_around:
                board[r][c] = player
                if has_winner(board, board_size, empty, 5) == player:
                    board[r][c] = empty
                    return (r, c)
                board[r][c] = empty
    return None

def check_for_immediate_win2(board, player, coords_around):
    """
    Returns (row, col) if the 'player' can create a empty color color color color empty pattern. Otherwise, returns None.
    """
    board_size = len(board)
    empty = 0
    
    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] == empty and (r, c) in coords_around:
                board[r][c] = player
                if has_4_row_empty(board, board_size, empty):
                    board[r][c] = empty
                    return (r, c)
                board[r][c] = empty
    return None

def has_4_row_empty(board, board_size, empty):
    def check_direction2(row, col, d_row, d_col):
        if (
            0 <= row + 4 * d_row < board_size and
            0 <= col + 4 * d_col < board_size
        ):
            if (
                board[row][col] == empty and
                board[row + d_row][col + d_col] == board[row + 2 * d_row][col + 2 * d_col] == 
                board[row + 3 * d_row][col + 3 * d_col] == board[row + 4 * d_row][col + 4 * d_col] != empty
            ):
                return board[row + d_row][col + d_col]
        return None

    for row in range(board_size):
        for col in range(board_size):
            for d_row, d_col in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                result = check_direction2(row, col, d_row, d_col)
                if result is not None:
                    return result
    return None

def check_for_immediate_win3(board, player, coords_around):
    """
    Returns (row, col) if there are more than 2 patterns in this board.
    """
    board_size = len(board)
    empty = 0
    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] == empty and (r, c) in coords_around:
                # Simulate the player's move
                board[r][c] = player
                # Generate patterns
                pat = patterns(player)
                # Count the patterns
                pattern_counts = count_patterns(board, pat)
                # Analyze patterns
                if analyse_pats(pattern_counts):
                    board[r][c] = empty  # Undo the move
                    return (r, c)  # Immediate win found
                board[r][c] = empty  # Undo the move
    return None  # No immediate win found

def patterns(player):
    """
    Returns the list of patterns to look for in the board.
    """
    if player==1:
        pat1 = [
            #'01011', '11100', '00111',
            #'10110', '01101', '11010',
            '01110'
        ]
        pat2 = ['10111', '11110', '11011', '01111', '11101']
    else: 
        pat1 = [
            #'01011', '11100', '00111',
            #'10110', '01101', '11010',
            '02220'
        ]
        pat2 = ['20222', '22220', '22022', '02222', '22202']
    return pat1 + pat2

def count_patterns(board, patterns):
    """
    Counts the occurrences of each pattern in the board using a sliding window approach.
    """
    pattern_counts = {pattern: 0 for pattern in patterns}

    # Check rows
    for row in board:
        row_str = ''.join(map(str, row))
        for pattern in patterns:
            pattern_counts[pattern] += sliding_window_match(row_str, pattern)

    # Check columns
    for col in board.T:
        col_str = ''.join(map(str, col))
        for pattern in patterns:
            pattern_counts[pattern] += sliding_window_match(col_str, pattern)

    # Check diagonals (top-left to bottom-right)
    for diag in range(-board.shape[0] + 1, board.shape[1]):
        diagonal = np.diagonal(board, offset=diag)
        diagonal_str = ''.join(map(str, diagonal))
        for pattern in patterns:
            pattern_counts[pattern] += sliding_window_match(diagonal_str, pattern)

    # Check anti-diagonals (top-right to bottom-left)
    flipped_board = np.fliplr(board)
    for diag in range(-flipped_board.shape[0] + 1, flipped_board.shape[1]):
        diagonal = np.diagonal(flipped_board, offset=diag)
        diagonal_str = ''.join(map(str, diagonal))
        for pattern in patterns:
            pattern_counts[pattern] += sliding_window_match(diagonal_str, pattern)

    return pattern_counts

def sliding_window_match(line, pattern):
    """
    Matches a pattern in a string using a sliding window approach.
    """
    matches = 0
    for i in range(len(line) - len(pattern) + 1):
        if line[i:i + len(pattern)] == pattern:
            matches += 1
    return matches

def analyse_pats(pattern_counts):
    """
    Analyze the patterns and decide whether to play the move.
    """
    total_count = sum(pattern_counts.values())
    if total_count >= 2:
        print("Play the move!")
        print(f"Patterns: {list(pattern_counts.keys())}")
        print(f"Counts: {list(pattern_counts.values())}")
        return True
    else:
        return False


def get_coords_around(board_size, board, ranger=1):
    """
    Finds potential moves by identifying the empty cells adjacent (including diagonals)
    to any occupied cell. This restricts the search space significantly.
    """
    # Occupied cells
    occupied = np.nonzero(board)
    
    potential_dict = {}
    for y, x in zip(occupied[0], occupied[1]):
        # For each neighbor within the range
        for dy in range(-ranger, ranger + 1):
            for dx in range(-ranger, ranger + 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < board_size and 0 <= nx < board_size:
                    if board[ny][nx] == 0:  # empty cell
                        potential_dict[(nx, ny)] = 1

    if not potential_dict:
        # If no neighbors found, fallback on all empty cells (rare but can happen at start).
        empty_positions = np.argwhere(board == 0)
        candidate_cols = [pos[1] for pos in empty_positions]
        candidate_rows = [pos[0] for pos in empty_positions]
    else:
        candidate_cols = [key[0] for key in potential_dict]
        candidate_rows = [key[1] for key in potential_dict]

    return candidate_cols, candidate_rows


def minimax(board, depth, alpha, beta, is_maximizing_player, ai_player, start_time, timeout):
    """
    Standard minimax with alpha-beta pruning.
      - depth: how deep we still search
      - alpha, beta: bounds for alpha-beta
      - is_maximizing_player: True if we are maximizing the AI's advantage
      - ai_player: the AI's piece (1 for black, 2 for white)
    """
    if time.time() - start_time > timeout: # Return a neutral score on timeout
        return 0
    
    score = evaluate_board(board, ai_player)

    # Large positive/negative scores mean game over or near game over
    # Also if we've reached the cutoff depth, return the evaluation.
    if abs(score) >= 20000000 or depth == 0:
        return score

    opp_player = 2 if ai_player == 1 else 1

    board_size = len(board)
    candidate_cols, candidate_rows = get_coords_around(board_size, board)

    if is_maximizing_player:
        best_val = float("-inf")
        for cx, cy in zip(candidate_cols, candidate_rows):
            if board[cy][cx] == 0:
                board[cy][cx] = ai_player
                val = minimax(board, depth - 1, alpha, beta, False, ai_player, start_time,timeout)
                board[cy][cx] = 0
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
        return best_val
    else:
        best_val = float("inf")
        for cx, cy in zip(candidate_cols, candidate_rows):
            if board[cy][cx] == 0:
                board[cy][cx] = opp_player
                val = minimax(board, depth - 1, alpha, beta, True, ai_player, start_time,timeout)
                board[cy][cx] = 0
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
        return best_val


def evaluate_board(board, ai_player):
    """
    Evaluate the board to produce a heuristic score from the perspective of ai_player.
    A positive score is better for ai_player, negative is better for the opponent.
    """
    patterns = {
        "11111": 30000000,
        "22222": -30000000,

        "011110": 20000000, 
        "022220": -20000000,
        
        "011112": 50000,
        "211110": 50000,
        "022221": -50000,
        "122220": -50000,

        "01110": 30000,
        "02220": -30000,
        '011010': 15000,
        '010110': 15000,
        '022020': -15000,
        '020220': -15000,
        '001112': 2000,
        '211100': 2000,
        '002221': -2000,
        '122200': -2000,
        '211010': 2000,
        '210110': 2000,
        '010112': 2000,
        '011012': 2000,
        '122020': -2000,
        '120220': -2000,
        '020221': -2000,
        '022021': -2000,
        '01100': 500,
        '00110': 500,
        '02200': -500,
        '00220': -500
    }
    val = 0
    lines = board_to_strings(board, ai_player)
    for line in lines:
        length_of_line = len(line)
        max_pat_len = 6
        for start_idx in range(length_of_line):
            for pat_len in range(5, max_pat_len + 1):
                end_idx = start_idx + pat_len
                if end_idx > length_of_line:
                    break
                substring = line[start_idx:end_idx]
                if substring in patterns:
                    val += patterns[substring]
    return val


def board_to_strings(board, ai_player):
    """
    Convert the board into an array of strings representing rows, columns, and diagonals.
    """
    def cell_to_char(cell):
        if cell == 0:
            return "0"
        elif cell == ai_player:
            return "1"
        else:
            return "2"

    size = len(board)
    row_list = []
    col_list = []
    diag_list = []

    # Rows
    for row in board:
        row_str = "".join(cell_to_char(c) for c in row)
        row_list.append(row_str)
    # Columns
    for col_idx in range(size):
        col_str = "".join(cell_to_char(board[row_idx][col_idx]) for row_idx in range(size))
        col_list.append(col_str)
    # Diagonals (left-to-right)
    bdiag = [board.diagonal(i) for i in range(-size + 1, size)]
    # Diagonals (right-to-left)
    fdiag = [np.fliplr(board).diagonal(i) for i in range(-size + 1, size)]
    for diag in bdiag:
        diag_list.append("".join(cell_to_char(c) for c in diag))
    for diag in fdiag:
        diag_list.append("".join(cell_to_char(c) for c in diag))
    return row_list + col_list + diag_list


#%%End game functions

def has_winner(board, board_size, empty, win_length):
    """
    Checks if there is a winner by scanning in 4 directions (horizontal, vertical, 2 diagonals).
    Returns the piece number (1 or 2) that wins, or None if no winner yet.
    """
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] != empty:
                # 1) Horizontal
                if check_direction(board, row, col, 0, 1, win_length):
                    return board[row][col]
                # 2) Vertical
                if check_direction(board, row, col, 1, 0, win_length):
                    return board[row][col]
                # 3) Diagonal (down-right)
                if check_direction(board, row, col, 1, 1, win_length):
                    return board[row][col]
                # 4) Diagonal (up-right)
                if check_direction(board, row, col, -1, 1, win_length):
                    return board[row][col]
    return None

def check_direction(board, start_row, start_col, delta_row, delta_col, length):
    """
    From (start_row, start_col), move step by step with (delta_row, delta_col)
    and see if we get 'length' same pieces in a row.
    """
    player = board[start_row][start_col]
    for i in range(1, length):
        r = start_row + delta_row * i
        c = start_col + delta_col * i
        if not (0 <= r < len(board) and 0 <= c < len(board)):
            return False
        if board[r][c] != player:
            return False
    return True

def board_is_full(board, empty):
    return not np.any(board == empty)

#%%Extra functions 

def ai_vs_ai():
    """
    A function allowing the AI to play itself to see where it struggles 
    """
    moves=3
    board_size, win_length, EMPTY, BLACK, WHITE = constants()
    board, center = initial_board(board_size)
    print("AI vs AI mode initiated! Press Enter to progress through each turn.")
    #can load a scenario to see how it performs from there
    scenarios = list_of_scenarios(BLACK, WHITE)
    scenario = load_board(board_size, scenarios)
    if scenario:
        board = scenario["board"]
        player_symbol = scenario["player_symbol"]
        ai1=player_symbol
        ai_symbol = scenario["ai_symbol"]
        ai2=ai_symbol
        player_turn = scenario["player_turn"]
        ai1_turn=player_turn
        current_player = scenario["current_player"]
    else:
        # Default to an empty board if no scenario was loaded
        ai1=BLACK
        ai2=WHITE
        ai1_turn=True
        current_player=BLACK
        board=random_move(board)
        
    print_board(board, board_size)
    print()
    black_second_move=True
    
    while True:
        print(f"AI {2 if current_player == WHITE else 1}'s turn! Press Enter to continue.")
        input()
        if ai1_turn:
            print(f'move : {moves}')
            if black_second_move==True:
                row,col=7,3
                start_time=0
                end_time=0
                black_second_move=False
            else:
                start_time = time.time()
                row, col = find_best_move(
                    board=board,
                    current_player=current_player,
                    board_size=board_size,
                )
                end_time = time.time()
        else:
            print(f'move : {moves}')
            start_time = time.time()  
            row, col = find_best_move(
                board=board,
                current_player=current_player,
                board_size=board_size,
            )
            end_time = time.time()
        if row != -1 and col != -1:
            board[row][col] = current_player
            elapsed_time = end_time - start_time
            print(f"The AI took {elapsed_time:.2f} seconds")
            print(f"AI played at {chr(65 + row)}{col}.")
        print_board(board, board_size)
        winner = has_winner(board, board_size, EMPTY, win_length)
        if winner==ai1:
            the_winner='ai1 wins'
        else:
            the_winner='ai2 wins'
        if winner:
            print(f"{the_winner}!")
            break
        if board_is_full(board, EMPTY):
            print("It's a draw!")
            break
        ai1 = not ai1
        current_player = WHITE if current_player == BLACK else BLACK
        moves+=1
    
def random_move(board):
    """
    Generate a random move for white second move otherwise plays same game every time 
    """
    x = random.randint(0, 14)
    y = random.randint(0, 14)
    board[x][y]=2
    return board
    
def load_board(board_size, scenarios):    
    """
    Load a predefined board scenario based on user input.
    """
    print("Available scenarios:")
    for idx, scenario in enumerate(scenarios.keys(), start=1):
        print(f"{idx}. {scenario}")
    choice = input("Enter the number of the scenario you want to load (or press Enter to skip): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(scenarios):
        scenario_name = list(scenarios.keys())[int(choice) - 1]
        scenario = scenarios[scenario_name]
        print(f"Loaded scenario: {scenario_name}")
        return scenario
    else:
        print("Invalid choice or no scenario selected. Starting with an empty board.")
        return None

def list_of_scenarios(BLACK, WHITE):
    """
    Define scenarios with correct board and symbol mappings.
    """
    scenarios = {
        "Scenario 1": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 2,
            "ai_symbol": 1,
            "player_turn": False,
            "current_player": 1
        },
        #Unbeatable double threat scenario
        "Scenario 2": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 2,
            "ai_symbol": 1,
            "player_turn": False,
            "current_player": 1
        },
        #Possible double threat but AI immediatly blocks 
        "Scenario 3": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 1,
            "ai_symbol": 2,
            "player_turn": False,
            "current_player": 2
        },
        #AI can create a double threat can AI see it?
        "Scenario 4": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 1,
            "ai_symbol": 2,
            "player_turn": False,
            "current_player": 2
        },
        "Scenario 5": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 1,
            "ai_symbol": 2,
            "player_turn": False,
            "current_player": 2
        },
        "Scenario 6": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 1,
            "ai_symbol": 2,
            "player_turn": True,
            "current_player": 1
        },
        "Scenario 7": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 2,
            "ai_symbol": 1,
            "player_turn": False,
            "current_player": 1
        },
        "Scenario 8": { #double attack possibility
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 2,
            "ai_symbol": 1,
            "player_turn": False,
            "current_player": 1
        },
        "Scenario 9": { #double attack defense
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 1,
            "ai_symbol": 2,
            "player_turn": False,
            "current_player": 2
        },
        "Scenario 10": { #double attack defense
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 1,
            "ai_symbol": 2,
            "player_turn": False,
            "current_player": 2
        },
        "Scenario 11": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 2,
            "ai_symbol": 1,
            "player_turn": False,
            "current_player": 1
        },
        #rather have AI not complete the win in certain scenarios like 12 rather then 
        #play a get 4 in a row 
        "Scenario 12": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 1,
            "ai_symbol": 2,
            "player_turn": False,
            "current_player": 2
        },
        "Scenario 13": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 1,
            "ai_symbol": 2,
            "player_turn": False,
            "current_player": 2
        },
        "Scenario 14": {
            "board": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            "player_symbol": 1,
            "ai_symbol": 2,
            "player_turn": False,
            "current_player": 2
        },
    }
    return scenarios

def run_scenario():
    """
    Run the game with the selected or default scenario 
    """
    moves=1
    black_second_move = False
    
    # --- 1. Load constants
    board_size, win_length, EMPTY, BLACK, WHITE = constants()
    BLACK, WHITE = 1, 2
    
    # --- 2. Load scenarios and board
    scenarios = list_of_scenarios(BLACK, WHITE)
    scenario = load_board(board_size, scenarios)
    
    if scenario:
        board = scenario["board"]
        player_symbol = scenario["player_symbol"]
        ai_symbol = scenario["ai_symbol"]
        player_turn = scenario["player_turn"]
        current_player = scenario["current_player"]
    else:
        # Default to an empty board if no scenario was loaded
        board = np.full((board_size, board_size), EMPTY)
        player_symbol, ai_symbol, player_turn = choose_side(BLACK, WHITE)
        current_player = BLACK if player_turn else WHITE
    
    # --- 3. Display the board
    print_board(board, board_size)
    print()
    
    while True:
        print(f'move: {moves}')
        if player_turn:
            # --- Human move
            print("Your turn!")
            move = input("Enter your move (e.g., H7): ").strip().upper()
            try:
                row, col = ord(move[0]) - 65, int(move[1:])
                if is_valid_move(board, row, col, board_size, EMPTY):
                    board[row][col] = current_player
                else:
                    print("Invalid move. Try again.")
                    continue
            except:
                print("Invalid format. Try again.")
                continue
            if player_symbol == BLACK:
                black_second_move = False
        else:
            # --- AI move
            print(ai_symbol)
            print("AI's turn!")
            start_time = time.time()
            row, col = find_best_move(board, current_player, board_size)
            end_time = time.time()
    
            if row != -1 and col != -1:
                board[row][col] = current_player
                elapsed_time = end_time - start_time
                print(f"The AI took {elapsed_time:.2f} seconds")
                print(f"AI played at {chr(65 + row)}{col}.")
    
        # --- 4. Print board and check for winner or draw
        print_board(board, board_size)
        winner = has_winner(board, board_size, EMPTY, win_length)
        if winner:
            if winner == ai_symbol:
                print("AI wins!")
            else:
                print("You win!")
            break
    
        if board_is_full(board, EMPTY):
            print("It's a draw!")
            break
    
        # --- 5. Swap turns
        player_turn = not player_turn
        current_player = WHITE if current_player == BLACK else BLACK    
        moves+=1

