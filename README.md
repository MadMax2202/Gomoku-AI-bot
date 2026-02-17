# Gomoku-AI-bot



\\## Introduction



The Gomoku AI game script is an implementation of a competitive board game where a human player competes against an AI opponent. It combines board representation, interactive gameplay, AI strategies, and validation of game rules.







Below, the code is dissected into its primary components, each explained in detail, followed by an analysis of its architectural design and strategic implications.







---







\\## Table of Contents



\\- \\\[1. Game Initialization and Setup](#1-game-initialization-and-setup)



\&nbsp; - \\\[1.1 Constants and Rules](#11-constants-and-rules)



\&nbsp; - \\\[1.2 Board Initialization](#12-board-initialization)



\&nbsp; - \\\[1.3 Player Configuration](#13-player-configuration)



\&nbsp; - \\\[1.4 Print Board](#14-print-board)



\\- \\\[2. Core Gameplay Loop](#2-core-gameplay-loop)



\&nbsp; - \\\[2.1 Player Moves](#21-player-moves)



\&nbsp; - \\\[2.2 Black’s Second Move Restriction](#22-blacks-second-move-restriction)



\&nbsp; - \\\[2.3 AI Moves](#23-ai-moves)



\\- \\\[3. AI Decision-Making](#3-ai-decision-making)



\&nbsp; - \\\[3.1 Limit Search Space](#31-limit-search-space)



\&nbsp; - \\\[3.2 Immediate Responses](#32-immediate-responses)



\&nbsp;   - \\\[3.2.1 Immediate Win or Block](#321-immediate-win-or-block)



\&nbsp;   - \\\[3.2.2 Play or Prevent 3 Pieces with Empty Ends](#322-play-or-prevent-3-pieces-with-empty-ends)



\&nbsp;   - \\\[3.2.3 Play or Prevent a Double Attack](#323-play-or-prevent-a-double-attack)



\&nbsp; - \\\[3.3 Heuristic Evaluation](#33-heuristic-evaluation)



\&nbsp; - \\\[3.4 Minimax Algorithm with Alpha-Beta Pruning](#34-minimax-algorithm-with-alpha-beta-pruning)



\&nbsp;   - \\\[3.4.1 Alpha-Beta Pruning](#341-alpha-beta-pruning)



\&nbsp;   - \\\[3.4.2 Iterative Deepening](#342-iterative-deepening)



\&nbsp;   - \\\[3.4.3 Integration of evaluate\\\_board as a Cost Function](#343-integration-of-evaluate\\\_board-as-a-cost-function)



\&nbsp;   - \\\[3.4.4 Sliding Window Approach](#344-sliding-window-approach)



\\- \\\[4. Extra Functions](#4-extra-functions)



\&nbsp; - \\\[4.1 AI vs AI](#41-ai-vs-ai)



\&nbsp; - \\\[4.2 Run Scenario](#42-run-scenario)



\\- \\\[5. Possible Improvements](#5-possible-improvements)



\&nbsp; - \\\[5.1 Consolidating Immediate Checks](#51-consolidating-immediate-checks)



\&nbsp; - \\\[5.2 Enhancing the Cost Function](#52-enhancing-the-cost-function)



\&nbsp; - \\\[5.3 Optimizing Search Space Reduction](#53-optimizing-search-space-reduction)



\&nbsp; - \\\[5.4 Iterative Deepening Refinements](#54-iterative-deepening-refinements)



\&nbsp; - \\\[5.5 Improved Data Structures](#55-improved-data-structures)







---







\\## 1. Game Initialization and Setup







\\### 1.1 Constants and Rules



The `constants` function defines the critical parameters for the game. These constants ensure that the rules and dimensions of the game are consistent and scalable. For instance, by changing `win\\\_length` or `board\\\_size`, the game could support variations like larger boards or longer win conditions.







\\### 1.2 Board Initialization



The `initial\\\_board` function creates the initial state of the game board. A 15x15 NumPy array is initialized with zeros (`EMPTY`), representing an empty board. A black piece (1) is placed at the center of the board, aligning with the rule that black always plays first at the center.







NumPy’s array manipulation capabilities allow for efficient board representation, which is crucial for AI evaluations and game logic.







\\### 1.3 Player Configuration



The `choose\\\_side` function allows the human player to select their side:



\\- If the player chooses black (X), the AI is assigned white (O) and vice versa.



\\- The function also determines the starting turn, ensuring that black plays first.







The chosen configuration directly affects the flow of the game, as the AI must adjust its strategies based on whether it plays first or second.







\\### 1.4 Print Board



The `print\\\_board` function plays a vital role in the user experience by providing a clear and readable visual representation of the game board. It is designed to display the current state of the board after each turn, enabling the player to make informed decisions.







---







\\## 2. Core Gameplay Loop



The game proceeds through a loop that alternates between the human player and the AI until a winner is determined or the board is filled.







\\### 2.1 Player Moves



When it is the player’s turn, the script prompts them to input a move in the format `H7`, representing row H and column 7. The input is processed as follows:







1\\. \\\*\\\*Parsing\\\*\\\*



\&nbsp;  - The row letter is converted to an integer index using ASCII values (`ord(move\\\[0]) - 65`)



\&nbsp;  - The column number is parsed directly.







2\\. \\\*\\\*Validation\\\*\\\*



\&nbsp;  The `is\\\_valid\\\_move` function ensures that:



\&nbsp;  - The move is within the board’s boundaries (`0 <= row < board\\\_size` and `0 <= col < board\\\_size`)



\&nbsp;  - The targeted cell is empty (`board\\\[row]\\\[col] == EMPTY`)







If the move is invalid, the player receives an error message and is prompted to try again. This validation process ensures the integrity of the game state and prevents illegal moves.







\\### 2.2 Black’s Second Move Restriction



This rule is enforced using the `restricted\\\_zone` variable, which defines the boundaries of the restricted area. The validation logic ensures compliance with this rule.







\\### 2.3 AI Moves



When it is the AI’s turn, the `find\\\_best\\\_move` function determines its move. The AI’s decision-making process involves heuristics, pattern recognition, and the minimax algorithm with alpha-beta pruning (discussed in Section 3).







---







\\## 3. AI Decision-Making



The AI in this script is designed to emulate intelligent gameplay by combining multiple strategies and computational techniques.







\\### 3.1 Limit Search Space



The `get\\\_coords\\\_around` function is pivotal for optimizing the AI's decision-making process by narrowing down the search space for potential moves to strategically relevant areas. It dynamically adjusts the search range based on the context, enabling the AI to evaluate moves more efficiently.







Specifically:



\\- \\\*\\\*For Immediate Responses:\\\*\\\* the function looks two spaces in advance around occupied cells (`ranger=2`). This extended range ensures that the AI considers moves not only adjacent to existing pieces but also those slightly farther away. This is critical for detecting immediate threats (e.g., an opponent's four-in-a-row setup) or opportunities (e.g., extending a three-in-a-row pattern).



\\- \\\*\\\*For Minimax Evaluations:\\\*\\\* during deeper exploration with minimax, the function reduces its range to one space in advance (`ranger=1`). This tighter scope focuses the AI on immediate clusters of activity, where moves are most likely to impact the game state.







\\### 3.2 Immediate Responses



Before engaging in deeper analysis, the AI checks for urgent scenarios.







\\#### 3.2.1 Immediate Win or Block



The `check\\\_for\\\_immediate\\\_win` function scans the board for opportunities where the AI can win in the current turn by simulating moves across all empty cells:







1\\. \\\*\\\*Simulation\\\*\\\*



\&nbsp;  - Iterate over every empty cell.



\&nbsp;  - Temporarily place the AI’s piece (1 or 2) and check if the move results in five consecutive pieces in any direction.







2\\. \\\*\\\*Validation\\\*\\\*



\&nbsp;  - Call `has\\\_winner` to determine if the simulated move results in a win.



\&nbsp;  - If a win is detected, return the move immediately.







Similarly, the AI checks if the player is one move away from winning (same logic), ensuring it can block imminent threats.







\\#### 3.2.2 Play or Prevent 3 Pieces with Empty Ends



The AI checks for scenarios where three consecutive pieces are flanked by empty cells (e.g., `011100` or `022200`). These patterns are critical because:



\\- \\\*\\\*Offensively:\\\*\\\* the AI can extend its row to create a four-in-a-row or set up a future win.



\\- \\\*\\\*Defensively:\\\*\\\* blocking the player’s formation prevents the opponent from gaining momentum.







The `check\\\_for\\\_immediate\\\_win2` function identifies these patterns and either extends the AI’s row or blocks the player’s.







\\#### 3.2.3 Play or Prevent a Double Attack



A double attack occurs when a move creates two simultaneous threats, forcing the opponent into a defensive position. For example:



\\- Placing a piece can create two separate four-in-a-row patterns that require blocking.



\\- The AI actively looks for opportunities to create double attacks (`check\\\_for\\\_immediate\\\_win3`) and prioritizes these moves.







Conversely, if the player is poised to execute a double attack, the AI preemptively blocks the setup.







These immediate checks ensure that the AI can handle critical situations without unnecessary computations, maintaining its competitive edge.







\\### 3.3 Heuristic Evaluation



The `quick\\\_heuristic` function assigns scores to potential moves based on their proximity to existing pieces of the same color. This heuristic is particularly effective for:



\\- Prioritizing moves near established rows of pieces.



\\- Expanding clusters of pieces to maximize offensive potential.







By using heuristics, the AI narrows down the search space and focuses on high-value moves. This ordering is crucial because alpha-beta pruning works best when the most promising moves are evaluated earlier in the search.







\\### 3.4 Minimax Algorithm with Alpha-Beta Pruning



The minimax algorithm simulates and evaluates potential future moves by alternating objectives:



\\- \\\*\\\*Maximizing\\\*\\\* the AI’s advantage (AI’s turn)



\\- \\\*\\\*Minimizing\\\*\\\* the player’s advantage (predicting the player’s counter-moves)







This builds a decision tree where:



\\- Each node is a game state resulting from a move.



\\- Each branch is a possible move from that state.







The algorithm evaluates game states using a cost function (`evaluate\\\_board`) and selects the move that yields the most favorable outcome.







\\#### 3.4.1 Alpha-Beta Pruning



Alpha-beta pruning eliminates unnecessary branches that cannot affect the outcome by maintaining:



\\- \\\*\\\*Alpha:\\\*\\\* best score the maximizing player (AI) can guarantee



\\- \\\*\\\*Beta:\\\*\\\* best score the minimizing player (human) can guarantee







When a branch cannot surpass these bounds, the algorithm prunes (skips) further exploration, reducing computational overhead.







\\#### 3.4.2 Iterative Deepening



The AI uses iterative deepening by exploring deeper levels incrementally until a time constraint (e.g., five seconds) is reached. This guarantees a valid move within the time limit even if the full search is not complete.







\\#### 3.4.3 Integration of evaluate\\\_board as a Cost Function



The `evaluate\\\_board` function assigns a numerical score to a board state based on:







1\\. \\\*\\\*Offensive patterns\\\*\\\*



\&nbsp;  - `11111`: five in a row (immediate win) → very high positive score (e.g., `+30000000`)



\&nbsp;  - `011110`: open four (near win) → high positive score (e.g., `+20000000`)







2\\. \\\*\\\*Defensive patterns\\\*\\\*



\&nbsp;  - `022220`: opponent open four → very negative score (e.g., `-30000000`)



\&nbsp;  - `002220`: opponent threatening setup → heavily penalized (e.g., `-200000`)







By assigning weighted scores to these patterns, `evaluate\\\_board` quantifies the strategic value of a game state. Minimax uses this to decide which branches are favorable.







\\#### 3.4.4 Sliding Window Approach



To detect patterns efficiently, `evaluate\\\_board` uses a sliding window technique:



1\\. Scan the board row-by-row, column-by-column, and along both diagonals.



2\\. Slide a window of 5–6 cells (depending on the pattern) across sequences.



3\\. Compare each substring in the window to predefined patterns.



4\\. Score matches based on strategic value.







---







\\## 4. Extra Functions



The extra functions were critical tools for testing, refining, and enhancing the AI’s strategic capabilities. They allowed controlled experimentation with specific scenarios and streamlined debugging by simulating various configurations and interactions.







Through targeted testing, the extra functions provided insights into the AI’s decision-making process, highlighting areas for improvement and verifying responses to complex scenarios.







\\### 4.1 AI vs AI



The `ai\\\_vs\\\_ai` function allows two AI instances to compete against each other, alternating turns and making moves based on their strategies. By observing outcomes, developers can:



\\- Identify weak points



\\- Evaluate strategy adaptability



\\- Measure performance







The function also supports loading predefined scenarios or initializing an empty board (e.g., analyzing double-threat resolution or assessing potential wins).







\\### 4.2 Run Scenario



The `run\\\_scenario` function integrates scenario loading into an interactive game loop to test the AI directly in a human-vs-AI context.







\\\*\\\*Key features:\\\*\\\*



1\\. \\\*\\\*Scenario loading:\\\*\\\* users choose from predefined scenarios (e.g., Scenario 2 tests blocking a double threat).



2\\. \\\*\\\*Real-time interaction:\\\*\\\* players move and observe AI responses dynamically.



3\\. \\\*\\\*Validation and debugging:\\\*\\\* validates moves and detects winners/draws/board completion to debug edge cases.







---







\\## 5. Possible Improvements



Although the Gomoku AI exhibits decision-making and adaptability, several enhancements could improve efficiency, scalability, and strategic depth.







\\### 5.1 Consolidating Immediate Checks



The AI currently performs multiple immediate checks (win, block, double attack, etc.) across several functions, which introduces redundancy and overhead.







An improvement would be consolidating these into a single function or embedding them into `evaluate\\\_board`, which could:



\\- Eliminate repetitive code



\\- Reduce time complexity from repeated scans



\\- Make decision-making more cohesive by integrating immediate responses into heuristic evaluation







\\### 5.2 Enhancing the Cost Function



`evaluate\\\_board` is the cornerstone of decision-making. It could be improved by:



1\\. \\\*\\\*Dynamic pattern scoring:\\\*\\\* adjust scores based on context (e.g., a double threat might be worth more if it blocks a critical opponent move).



2\\. \\\*\\\*Probabilistic models:\\\*\\\* predict the likelihood of certain moves leading to a win to support long-term strategies rather than only immediate gains.







This would enhance optimal move selection and reduce reliance on static predefined patterns.







\\### 5.3 Optimizing Search Space Reduction



`get\\\_coords\\\_around` could be improved with an adaptive search radius instead of a fixed range:



\\- Early game: wider radius for exploratory moves



\\- Late game: narrower radius to focus on critical areas







\\### 5.4 Iterative Deepening Refinements



Possible upgrades:



\\- \\\*\\\*Weighted depth exploration:\\\*\\\* explore higher-value branches deeper and prune lower-value ones earlier



\\- \\\*\\\*Early termination heuristics:\\\*\\\* if a win or catastrophic loss is detected early, stop exploring to save time







\\### 5.5 Improved Data Structures



Current implementation uses NumPy arrays. Alternatives that could improve performance:



1\\. \\\*\\\*Bitboards:\\\*\\\* binary representation of positions to speed up pattern recognition and validation



2\\. \\\*\\\*Sparse matrices:\\\*\\\* for larger boards with sparse activity to save memory and reduce unnecessary computation









