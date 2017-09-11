import random
import copy

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

#Stores info about puzzles    
class Puzzle:
    size = 0
    board = []

    #Constructor
    def __init__(self, b):
        self.size = len(b)
        self.board = b
        
    #Prints a puzzle
    def makeState(self):
        for row in self.board:
            print(row)
    
    # returns a board after a move is made
    # Takes a puzzle and move in the form (newSpaceRow, newSpaceCol, oldSpaceRow, oldSpaceCol)
    # Returns a board with the given move made
    def makeMove(self, move):
        newBoard = copy.deepcopy(self.board)
        newBoard[move[0]][move[1]] = "";
        newBoard[move[2]][move[3]] = self.board[move[0]][move[1]]
        return(newBoard)
    
    # generates all possible moves given a board
    # Returns a list of moves
    # A move is a tuple of the shape (newSpaceRow, newSpaceCol, oldSpaceRow, oldSpaceCol)
    def possMoves(self):
        for x in range(len(self.board)):
            if "" in self.board[x]:
                oldSpaceRow = x;
                spaceIndCol = self.board[x].index("")
                oldSpaceCol = spaceIndCol
                newPossInds = []
                if(oldSpaceRow - 1 >= 0):
                    newPossInds.append((oldSpaceRow-1, oldSpaceCol, oldSpaceRow, oldSpaceCol))
                if(oldSpaceRow + 1 < self.size):
                    newPossInds.append((oldSpaceRow+1, oldSpaceCol, oldSpaceRow, oldSpaceCol))
                if(oldSpaceCol - 1 >= 0):
                    newPossInds.append((oldSpaceRow, oldSpaceCol - 1, oldSpaceRow, oldSpaceCol))
                if(oldSpaceCol + 1 < self.size):
                    newPossInds.append((oldSpaceRow, oldSpaceCol + 1, oldSpaceRow, oldSpaceCol))
                return(newPossInds)
                break
            
    # Randomizes a puzzle from whatever state it's already in
    def randomBoard(self):
        numMoves = random.randrange(1,self.size * 1000)
        for x in range(numMoves):
            nextMoves = self.possMoves()
            moveInd = random.randrange(0,len(nextMoves))
            self.board = self.makeMove(nextMoves[moveInd])

def makeNode(state, parent, depth, pathCost):
    return [state, parent, depth, pathCost]

def generalSearch(queue, limit, numRuns):
    if queue == []:
        return False
    elif testProcedure(queue[0]):
        outputProcedure(numRuns, queue[0])
    elif limit == 0:
        print("Limit reached")
    else:
        limit -= 1
        numRuns += 1
        generalSearch(expandProcedure(queue[0], queue[1:len(queue)]), limit, numRuns)

p1 = Puzzle(makeGoalBoard(3))
p1.makeState()
print()
p1.randomBoard()
p1.makeState()
print()

p2 = Puzzle(makeGoalBoard(5))
p2.makeState()
print()
p2.randomBoard()
p2.makeState()
