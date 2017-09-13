import random
import copy

board = []
explored = []

# Creates a goal board based on a given size
# Takes an int called size
# Returns a goal board of given size
def makeGoalBoard(size):
    nBoard = []
    num = 1
    for x in range(size):
        nRow = []
        for y in range(size):
            if(num != size*size):
                nRow.append(num)
                num += 1
            else:
                nRow.append("")
        nBoard.append(nRow)
    return(nBoard)

#Prints a puzzle
def makeState(board):
    for row in board:
        print(row)

# returns a board after a move is made
# Takes a puzzle and move in the form (newSpaceRow, newSpaceCol, oldSpaceRow, oldSpaceCol)
# Returns a board with the given move made
def makeMove(board, move):
     newBoard = copy.deepcopy(board)
     newBoard[move[0]][move[1]] = "";
     newBoard[move[2]][move[3]] = board[move[0]][move[1]]
     return(newBoard)

# generates all possible moves given a board
# Returns a list of moves
# A move is a tuple of the shape (newSpaceRow, newSpaceCol, oldSpaceRow, oldSpaceCol)
def possMoves(board):
     for x in range(len(board)):
         if "" in board[x]:
            size = len(board);
            oldSpaceRow = x;
            spaceIndCol = board[x].index("")
            oldSpaceCol = spaceIndCol
            newPossInds = []
            if(oldSpaceRow - 1 >= 0):
                newPossInds.append((oldSpaceRow-1, oldSpaceCol, oldSpaceRow, oldSpaceCol))
            if(oldSpaceRow + 1 < size):
                newPossInds.append((oldSpaceRow+1, oldSpaceCol, oldSpaceRow, oldSpaceCol))
            if(oldSpaceCol - 1 >= 0):
                newPossInds.append((oldSpaceRow, oldSpaceCol - 1, oldSpaceRow, oldSpaceCol))
            if(oldSpaceCol + 1 < size):
                newPossInds.append((oldSpaceRow, oldSpaceCol + 1, oldSpaceRow, oldSpaceCol))
            return(newPossInds)
            break

# Randomizes a puzzle from whatever state it's already in
def randomBoard(board):
    randBoard = copy.deepcopy(board)
    numMoves = random.randrange(0, 19)
    for x in range(numMoves):
        nextPossMoves = possMoves(randBoard)
        nextMoveInt = random.randrange(0, len(nextPossMoves))
        randBoard = copy.deepcopy(makeMove(randBoard, nextPossMoves[nextMoveInt]))
    return randBoard

# Determines whether the board is in the goal state.
def testProcedure(currentNode):
    return currentNode[0] == makeGoalBoard(len(currentNode[0]))

# Follows and prints the path of the solution.
def outputProcedure(numRuns, currentNode):
    step = 0
    path = []
    while (numRuns >= 0 and currentNode is not None):
        path.append(currentNode[0])
        step += 1
        currentNode = currentNode[1]
        numRuns -= 1
    while (step > 0):
        makeState(path.pop())
        print()
        step -= 1

# Expands current node to all possible moves that have
# not already been explored.
# Returns a list of all possible successor nodes.
def expandProcedure(currentNode, queue, explored):
    possibleMoves = possMoves(currentNode[0])
    possibleMovesMade = []
    nqueue = copy.deepcopy(queue)

    # Finds the possible moves from a state.
    for move in possibleMoves:
        possibleMovesMade.append(makeMove(currentNode[0], move))
    # Finds possible states from the possible moves.
    for possibleMoveMade in possibleMovesMade:
        if (possibleMoveMade not in explored):
            nqueue.append(makeNode(possibleMoveMade, currentNode, currentNode[2] + 1, currentNode[3]))
    return nqueue

# Creates a node for the board state.
def makeNode(state, parent, depth, pathCost):
    return [state, parent, depth, pathCost]

# Runs a BFS to find the solution to a given board.
def generalSearch(queue, limit, numRuns):
    if queue == []:
        return False
    elif testProcedure(queue[0]):
        print("OUTPUT")
        outputProcedure(numRuns, queue[0])
    elif limit == 0:
        print("Limit reached")
    else:
        limit -= 1
        numRuns += 1
        node = queue[0]
        explored.append(node[0])
        generalSearch(expandProcedure(queue[0], queue[1:len(queue)], explored), limit, numRuns)

#p1 = makeGoalBoard(3)
#makeState(p1)
#print()
#randomBoard(p1)
#makeState(p1)
#print()

start = randomBoard(makeGoalBoard(3))
startNode = makeNode(start, None, 1, 0)
generalSearch([startNode], 1000, 0)


