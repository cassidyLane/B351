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



'''
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

def getLowestPrior(frontier):
    lowestPrior = frontier[0][0]
    lowestNode = frontier[0]

    for nodePrior in frontier:
        if nodePrior[0] < lowestPrior:
            lowestPrior = nodePrior[0]
            lowestNode = nodePrior

    return lowestNode

# Calculates how far each tile is from its goal state, and sums those distances
def heuristic(matrix, goal):
    sum = 0
    for i in range(0, len(goal)):
        for j in range(0, len(goal)):
            tile = goal[i][j]
            for k in range(0, len(matrix)):
                for l in range(0, len(matrix)):
                    if matrix[k][l] == tile:
                        sum += (k - i)*(k - i)+(j - l)*(j - l)
    return(sum)


def easyAStar(queue, limit, numRuns, heuristic, goalBoard):
    frontier = []
    exp = []

    frontier.append((0, queue[0]))

    while (len(frontier) > 0):
        current = getLowestPrior(frontier)
        frontier.remove(current)
        currentBoard = current[1][0]
        successors = expandProcedure(current[1], [], exp)

        for successor in successors:
            numRuns += 1
            limit -= 1

            if(testProcedure(successor[0], goalBoard)):
                print("OUTPUT")
                outputProcedure(numRuns, successor)
                return
            elif(limit == 0):
                print("LIMIT REACHED")
                return
            priority = successor[3] + heuristic(successor[0], goalBoard)

            addNode = True

            for x in range(len(frontier)):
                if(frontier[x][1][0] == successor[0]):
                    addNode = False
                    if(priority < frontier[x][0]):
                        addNode = True


            for x in range(len(exp)):
                if(exp[x][1][0] == successor[0]):
                    addNode = (False or addNode)
                    if(priority < exp[x][0]):
                        addNode = True

            if(addNode):
                frontier.append((priority, successor))
        exp.append(current)


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

#testUninformedSearch(randomBoard(makeGoalBoard(3)), makeGoalBoard(3), 1000)
#testInformedSearch(makeState(1,2,3,4,5,"",7,8,6), makeGoalBoard(3), 1000, heuristic)

#testUninformedSearch(makeState(1,2,6,3,5,"",4,7,8), makeGoalBoard(3), 1000)
#testInformedSearch(makeState(1,2,6,3,5,"",4,7,8), makeGoalBoard(3), 1000, heuristic)
#testInformedSearch(makeState(1,2,6,3,5,"",4,7,8), makeGoalBoard(3), 1000, heuristic)
#testInformedSearch(makeState(5,1,2,6,3,"",4,7,8), makeGoalBoard(3), 1000, heuristic)
testInformedSearch(makeState(3,5,6,1,4,8,"",7,2), makeGoalBoard(3), 1000, heuristic)
#testInformedSearch(makeState(8,7,6,5,4,3,2,1,""), makeGoalBoard(3), 1000, heuristic)


