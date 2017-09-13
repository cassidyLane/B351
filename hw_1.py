import random
import copy
import time

board = []
explored = []


# Creates a node for the board state.
def makeNode(state, parent, depth, pathCost):
    return [state, parent, depth, pathCost]

# Creates a state board based on given elements.
# Used for testing purposes.
def makeState(nw, n, ne, w, c, e, sw, s, se):
    row1 = [nw, n, ne]
    row2 = [w, c, e]
    row3 = [sw, s, se]
    return [row1, row2, row3]

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
def printState(board):
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
def testProcedure(currentState, goalState):
    return currentState == goalState

# Follows and prints the path of the solution.
def outputProcedure(numRuns, currentNode):
    step = 0
    path = []
    while (numRuns >= 0 and currentNode is not None):
        path.append(currentNode[0])
        step += 1
        currentNode = currentNode[1]
        numRuns -= 1
        print(step-1)
    while (step > 0):
        printState(path.pop())
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

# Runs a BFS to find the solution to a given board.
def generalSearch(queue, limit, numRuns, goal):
    if queue == []:
        return False
    elif testProcedure(queue[0][0], goal):
        print("OUTPUT")
        outputProcedure(numRuns, queue[0])
    elif limit == 0:
        print("Limit reached")
    else:
        limit -= 1
        numRuns += 1
        node = queue[0]
        explored.append(node[0])
        generalSearch(expandProcedure(queue[0], queue[1:len(queue)], explored), limit, numRuns, goal)

def distHeuristic(board):
    d = 0
    correctVal = 1
    realVal = 0
    for x in range(len(board)):
        for y in range(len(board)):
            if(board[x][y] == ""):
                realVal = len(board) * len(board)
            else:
                realVal = board[x][y]
            d += abs(correctVal-realVal)
            correctVal += 1
    return(int(3*d))

def numOutOfOrderHeuristic(board):
    outOfRow = 0
    outOfCol = 0
    flippedBoard = []
    modelBoard = makeGoalBoard(len(board))
    flippedModelBoard = []
    for x in range(len(board)):
        colAsRow = []
        modColAsRow = []
        for y in range(len(board)):
            colAsRow.append(board[y][x])
            modColAsRow.append(modelBoard[y][x])
        flippedBoard.append(colAsRow)
        flippedModelBoard.append(modColAsRow)
    for x in range(len(board)):
        for y in range(len(board[x])):
            if(board[x][y] not in modelBoard[x]):
                outOfRow += 1
            if(flippedBoard[x][y] not in flippedModelBoard[x]):
                outOfCol += 1
    return(outOfRow + outOfCol)

# Tests the uninformed search method.
def testUninformedSearch(init, goal, limit):
    initNode = makeNode(init, None, 0, 0)
    generalSearch([initNode], limit, 0, goal)

# Tests the informed search method.
# Takes a heuristic to determine which heuristic to use to calculate path cost
def testInformedSearch(init, goal, limit, heuristic):
    initNode = makeNode(init, None, 0, heuristic(init))
    #aStarSearch([initNode], limit, 0, goal, heuristic)

# Tests the functions created.
start = randomBoard(makeGoalBoard(3))
startNode = makeNode(start, None, 1, 0)

startTime = time.time()
testUninformedSearch(makeState(1,2,6,3,5,"",4,7,8), makeGoalBoard(3), 1000)
end = time.time()
print(end-startTime)

