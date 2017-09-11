    def search(self):
        self.parentTrace = {} #uses parent trace in the same way as DFS and BFS                                                                                                    

        fringe = PQ() # stores the fringe in a priority queue                                                                                                                      
        fringe.update(self.start,float('inf')) # puts the beginning of game in the fringe with highest priority                                                                    
        costs = {} # keeps track of the costs for each move                                                                                                                        
        costs[self.start.hash()] = 0 # this game state has a cost to reach of 0                                                                                                    
        while(not(fringe.isEmpty())):
            game = fringe.pop() # get the lowest priority game state                                                                                                               
            hash = game.hash()

            if game.isFinished():
                self.unwindPath()
                return

            successors = game.successors()

            for successor in successors:
                sGame = successor[0]
                sHash = sGame.hash()
                nCost = costs[hash] + 1 # adds 1 to the cost to get to this game state                                                                                             
                if((sHash not in costs) or (nCost < costs[sHash])): # if this game state isn't already in costs, or the newCost is lower than its cost in costs                    
                    costs[sHash] = nCost # add or replace the current cost for this game state to the new cost                                                                     
                    p = nCost + sGame.heuristic() # find what the priority should be by adding the cost to the heuristic                                                           
                    fringe.update(sGame, p) # put this game into the fringe                                                                                                        
                    self.parentTrace[sHash] = (hash, successor[1])
