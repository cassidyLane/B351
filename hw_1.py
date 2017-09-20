# Cassidy Wichowsky & Renee Bialas
# cwichows               rbialas

import random
import copy

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
            if (num != size * size):
                nRow.append(num)
                num += 1
            else:
                nRow.append("")
        nBoard.append(nRow)
    return (nBoard)

# Prints a puzzle.
def printState(board):
    for row in board:
        print(row)

# Takes a puzzle and move in the form (newSpaceRow, newSpaceCol, oldSpaceRow, oldSpaceCol)
# Returns a board with the given move made
def makeMove(board, move):
    newBoard = copy.deepcopy(board)
    newBoard[move[0]][move[1]] = "";
    newBoard[move[2]][move[3]] = board[move[0]][move[1]]
    return (newBoard)

# Generates all possible moves given a board.
# Returns a list of moves.
# A move is a tuple of the shape (newSpaceRow, newSpaceCol, oldSpaceRow, oldSpaceCol)
def possMoves(board):
    for x in range(len(board)):
        if "" in board[x]:
            size = len(board);
            oldSpaceRow = x;
            spaceIndCol = board[x].index("")
            oldSpaceCol = spaceIndCol
            newPossInds = []
            if (oldSpaceRow - 1 >= 0):
                newPossInds.append((oldSpaceRow - 1, oldSpaceCol, oldSpaceRow, oldSpaceCol))
            if (oldSpaceRow + 1 < size):
                newPossInds.append((oldSpaceRow + 1, oldSpaceCol, oldSpaceRow, oldSpaceCol))
            if (oldSpaceCol - 1 >= 0):
                newPossInds.append((oldSpaceRow, oldSpaceCol - 1, oldSpaceRow, oldSpaceCol))
            if (oldSpaceCol + 1 < size):
                newPossInds.append((oldSpaceRow, oldSpaceCol + 1, oldSpaceRow, oldSpaceCol))
            return (newPossInds)
            break

# Randomizes a puzzle from whatever state it's already in.
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
            nqueue.append(makeNode(possibleMoveMade, currentNode, currentNode[2] + 1, currentNode[3] + 1))
    return nqueue

# Runs a BFS to find the solution to a given board.
def generalSearch(queue, limit, numRuns, goal):
    if queue == []:
        return False
    elif testProcedure(queue[0][0], goal):
        print("OUTPUT:")
        outputProcedure(numRuns, queue[0])
        del explored[:]
    elif limit == 0:
        print("Limit reached")
        del explored[:]
    else:
        limit -= 1
        numRuns += 1
        node = queue[0]
        explored.append(node[0])
        generalSearch(expandProcedure(queue[0], queue[1:len(queue)], explored), limit, numRuns, goal)

# Calculates the Manhattan distance as a cost for each tile.
def numOutOfOrder(board, goalBoard):
    count = 0
    for x in range(len(board)):
        for y in range(len(board)):
            if board[x][y] != goalBoard[x][y] and board[x][y] != "":
                count += 1
    return count


'''
# Attempt at A* search with a heap.
def aStarHeapSearch(queue, limit, numRuns, heuristic, goalBoard):
    frontier = []
    #frontierDictionary = {}
    gameList = []
    priorityList = []
    nodeList = []

    queue[0][3] = 0 # Set path cost of first node to 0.

    heapq.heappush(frontier, (0, queue[0]))
    #frontierDictionary[queue[0][0]] = queue[0][3]
    gameList.append(queue[0][0])
    priorityList.append(0)
    nodeList.append(queue[0])



    while frontier:
        current = heapq.heappop(frontier)
        currentNode = current[1]
        aStarExplored.append(currentNode)
        cBoard = currentNode[0]
        cNodeInd = gameList.index(cBoard)
        del gameList[cNodeInd]
        del priorityList[cNodeInd]
        del nodeList[cNodeInd]

        if(testProcedure(cBoard, goalBoard)):
            print("OUTPUT:")
            outputProcedure(numRuns, currentNode)
            del aStarExplored[:] 
            return
        elif(limit == 0):
            print("Limit Reached")
            del aStarExplored[:]
            return

        successors = expandPQ(currentNode)

        for successor in successors:
            successorBoard = successor[0]
            successorCost = successor[3]
            priority = successorCost + heuristic(successorBoard)

            inGameList = False
            inExplored = False

            if successorBoard in gameList:
            #if frontierDictionary[successor[0]]:
                inGameList = True
                boardIndex = gameList.index(successorBoard)
                stolenPriority = priorityList[boardIndex]
                #stolenPriority = frontierDictionary[successor[0]]
                if priority < stolenPriority:
                    toRemove = nodeList[boardIndex]
                    frontier.remove((stolenPriority, toRemove))
                    heapq.heapify(frontier)
                    while(priority in priorityList):
                        priority += 1
                    heapq.heappush(frontier, (priority, successor))
                    priorityList[boardIndex] = priority
                    nodeList[boardIndex] = successor
                    #frontierDictionary.update(successor[0],priority)
                #else:
                    #frontierDictionary[successor[0]] = priority
            if (inLoT(successorBoard, aStarExplored)):
                print("in explored")
                inExplored = True
                exploredNode = searchLoT(successorBoard, aStarExplored)
                if priority < exploredNode[3]:
                    aStarExplored.remove(exploredNode)
                    heapq.heappush(frontier, (priority, successor))
                    gameList.append(exploredNode[0])
                    priorityList.append(priority)
                    nodeList.append(successor)

            if(not(inGameList or inExplored)):
                while(priority in priorityList):
                    priority += 1
                heapq.heappush(frontier, (priority, successor))
                gameList.append(successorBoard)
                priorityList.append(priority)
                nodeList.append(successor)
                #frontierDict[successor] = priority
                limit -= 1
                numRuns += 1
'''

# Returns the lowest priority node from frontier.
def getLowestPrior(frontier):
    lowestPrior = frontier[0][0]
    lowestNode = frontier[0]

    for nodePrior in frontier:
        if nodePrior[0] < lowestPrior:
            lowestPrior = nodePrior[0]
            lowestNode = nodePrior

    return lowestNode

# Calculates how far each tile is from its goal state, and sums those distances.
def heuristic(matrix, goal):
    sum = 0
    for i in range(0, len(goal)):
        for j in range(0, len(goal)):
            tile = goal[i][j]
            for k in range(0, len(matrix)):
                for l in range(0, len(matrix)):
                    if matrix[k][l] == tile:
                        sum += (k - i) * (k - i) + (j - l) * (j - l)
    return (sum)

# Implements A* search using a list.
# Solves a 8 Tile Sliding puzzle with heuristics.
def easyAStar(queue, limit, numRuns, heuristic, goalBoard):
    frontier = []
    exploredNodes = []

    frontier.append((0, queue[0]))

    while (len(frontier) > 0):
        currentNode = getLowestPrior(frontier)
        frontier.remove(currentNode)
        successors = expandProcedure(currentNode[1], [], exploredNodes)

        for successor in successors:

            if testProcedure(successor[0], goalBoard):
                print("OUTPUT: ")
                outputProcedure(numRuns, successor)
                return
            elif limit == 0:
                print("LIMIT REACHED")
                return
            priority = successor[3] + max(heuristic(successor[0], goalBoard), numOutOfOrder(successor[0], goalBoard))

            addedNode = True

            for x in range(len(frontier)):
                if (frontier[x][1][0] == successor[0]):
                    addedNode = False
                    if (priority < frontier[x][0]):
                        addedNode = True

            for x in range(len(exploredNodes)):
                if exploredNodes[x][1][0] == successor[0]:
                    addedNode = (False or addedNode)
                    if priority < exploredNodes[x][0]:
                        addedNode = True

            if addedNode:
                frontier.append((priority, successor))
                numRuns += 1
                limit -= 1
        exploredNodes.append(currentNode)


# Tests the uninformed search method.
def testUninformedSearch(init, goal, limit):
    initNode = makeNode(init, None, 0, 0)
    generalSearch([initNode], limit, 0, goal)

# Tests the informed search method.
# Takes a heuristic to determine which heuristic to use to calculate path cost
def testInformedSearch(init, goal, limit, heuristic):
    initNode = makeNode(init, None, 0, 0)
    easyAStar([initNode], limit, 0, heuristic, goal)

# Tests the functions created.
start = randomBoard(makeGoalBoard(3))
startNode = makeNode(start, None, 1, 0)



print("Uninformed")
# Completes 2 moves.
testUninformedSearch(makeState(1, 2, 3, 4, "", 5, 7, 8, 6), makeGoalBoard(3), 1000)
print()
# Completes 4 moves.
testUninformedSearch(makeState(1, 2, 3, 7, 4, 5, "", 8, 6), makeGoalBoard(3), 1000)
print()
# Completes 5 moves.
testUninformedSearch(makeState(1, 2, 3, 4, 8, "", 7, 6, 5), makeGoalBoard(3), 1000)
print()
# Completes 8 moves.
testUninformedSearch(makeState(4, 1, 3, 7, 2, 6, 5, 8, ""), makeGoalBoard(3), 1000)
print()
# Completes 9 moves.
testUninformedSearch(makeState(1, 6, 2, 5, 3, "", 4, 7, 8), makeGoalBoard(3), 1000)
print()

print("Informed")
# Completes 2 moves.
testInformedSearch(makeState(1, 2, 3, 4, "", 5, 7, 8, 6), makeGoalBoard(3), 1000, heuristic)
print()
# Completes 4 moves.
testInformedSearch(makeState(1, 2, 3, 7, 4, 5, "", 8, 6), makeGoalBoard(3), 1000, heuristic)
print()
# Completes 5 moves.
testInformedSearch(makeState(1, 2, 3, 4, 8, "", 7, 6, 5), makeGoalBoard(3), 1000, heuristic)
print()
# Completes 8 moves.
testInformedSearch(makeState(4, 1, 3, 7, 2, 6, 5, 8, ""), makeGoalBoard(3), 1000, heuristic)
print()
# Completes 9 moves.
testInformedSearch(makeState(1, 6, 2, 5, 3, "", 4, 7, 8), makeGoalBoard(3), 1000, heuristic)
print()
# Completes 11 moves.
testInformedSearch(makeState(5, 1, 2, 6, 3, "", 4, 7, 8), makeGoalBoard(3), 1000, heuristic)
print()