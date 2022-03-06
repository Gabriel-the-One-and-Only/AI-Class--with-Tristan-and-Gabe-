# multiAgents.py
# --------------
# This codebase is adapted from UC Berkeley AI. Please see the following information about the license.
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import sys

from sqlalchemy import null
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        newGhostPositions = successorGameState.getGhostPositions() #This does not return the actual moves of the ghosts just repeats the old positions; invalid

        scoreMod = 0 #score modifier that will be applied
        newFood = newFood.asList() #converts boolean grid to coordinates.
        currPos = currentGameState.getPacmanPosition()
        
        for ghost in currentGameState.getGhostPositions():
            print("This is the distance to the ghost: ", manhattanDistance(newPos, ghost))
            if manhattanDistance(newPos, ghost) < 2: # or newPos == currPos: #this line makes pacman avoid ghosts and optionally always move.
                scoreMod -= 1000 #discourages this idea
                #print("This code ran") debug line
                
        for food  in newFood:
            
            #first attempt
            """  if manhattanDistance(food,newPos) == 0: #This does not do anything
                scoreMod += 5
            elif manhattanDistance(newPos,food) < 3:
                scoreMod += 0.25
            elif manhattanDistance(newPos,food) < 6:
                scoreMod += 0.125
            elif manhattanDistance(newPos, food) > 5:
                scoreMod += 0.05 """
            scoreMod += 1/manhattanDistance(newPos, food) #prioritizes closer dots but still makes it aware of farther ones.
            
            
        #print (scoreMod, action, newFood) #debug lines
        #print(scoreMod)
        return successorGameState.getScore() + scoreMod

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #depth = 5
        #Declaring index number variable of last agent
        lastAgent = gameState.getNumAgents() - 1
        #Creating a recursive function to create the tree
        def minMax(depth, agent, treeGameState):
            #checking if any of the leaf nodes were reached
            if depth <= 0 or treeGameState.isWin() or treeGameState.isLose(): 
                #returns the score of game if this state was reached
                return [self.evaluationFunction(treeGameState),0]
            #This variable will hold the best action the agent can take and the score that would occur if all agents choose the optimal option
            bestValue = []
            #Checking all legal actions
            for action in treeGameState.getLegalActions(agent):
                #Going down the tree to the next agent. if the tree goes down 1 depth, the depth variable decriments    
                if(agent == lastAgent):
                    node = minMax(depth-1, 0, treeGameState.generateSuccessor(agent, action))
                else:
                    node = minMax(depth, agent+1, treeGameState.generateSuccessor(agent, action))
                #if this is the first cycle of the loop, set the first action to be the best one
                if bestValue == []:
                    bestValue = node
                #if the agent is pacman and the score is higher, bestvalue is replaced with the new score and the action it takes to get there
                if(agent == 0 and node[0] >= bestValue[0]):
                    bestValue[0] = node[0]
                    bestValue[1] = action
                #if the agent is a ghost and the score is lower, bestvalue is replaced with the new score and the action it takes to get there
                if(agent > 0 and node[0] <= bestValue[0]):
                    bestValue[0] = node[0]
                    bestValue[1] = action        
            return bestValue
                
        return minMax(self.depth, self.index, gameState)[1]

        
                    


               

                    


    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #Declaring index number variable of last agent
        lastAgent = gameState.getNumAgents() - 1
        #Creating a recursive function to create the tree
        def minMax(depth, agent, treeGameState, alpha, beta):
            #checking if any of the leaf nodes were reached
            if depth <= 0 or treeGameState.isWin() or treeGameState.isLose(): 
                #returns the score of game if this state was reached
                return [self.evaluationFunction(treeGameState),0]
            #This variable will hold the best action the agent can take and the score that would occur if all agents choose the optimal option
            #bestValue = [beta, 0]
            #worstValue = [alpha, 0]
            bestValue = [beta, 0]
            worstValue = [alpha, 0]
            #Checking all legal actions
            for action in treeGameState.getLegalActions(agent):
                #print("This is the best value:", bestValue)
                #print("This is the worst value:", worstValue)
                
                #Going down the tree to the next agent. if the tree goes down 1 depth, the depth variable decriments
                 
                if(agent == lastAgent):
                    node = minMax(depth-1, 0, treeGameState.generateSuccessor(agent, action), worstValue[0], bestValue[0])
                else:
                    node = minMax(depth, agent+1, treeGameState.generateSuccessor(agent, action), worstValue[0], bestValue[0])
                #if this is the first cycle of the loop, set the first action to be the best one and worst one
                if bestValue == [beta, 0]: #[beta, 0]:
                    #print("This code ran")
                    bestValue = node
                if worstValue == [alpha, 0]: #[alpha, 0]:
                    #print("This code ran")
                    worstValue = node
                #if the agent is pacman and the score is higher, bestvalue is replaced with the new score and the action it takes to get there
                if(agent == 0 and node[0] >= bestValue[0]):
                    bestValue[0] = node[0]
                    bestValue[1] = action
                    if(bestValue[0] > beta):
                        #print("Something was pruned")
                        return bestValue
                #if the agent is a ghost and the score is lower, worstValue is replaced with the new score and the action it takes to get there
                if(agent > 0 and node[0] <= worstValue[0]):
                    worstValue[0] = node[0]
                    worstValue[1] = action 
                    if(worstValue[0] < alpha):
                        #print("Something was pruned")
                        return worstValue
            if(agent == 0):      
                return bestValue
            else:
                return worstValue
                
        return minMax(self.depth, self.index, gameState, -sys.maxsize-1, sys.maxsize)[1] #start loop with alpha and beta at max and min representable numbers


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #Declaring index number variable of last agent
        lastAgent = gameState.getNumAgents() - 1
        #Creating a recursive function to create the tree
        def expecMax(depth, agent, treeGameState):
            #checking if any of the leaf nodes were reached
            if depth <= 0 or treeGameState.isWin() or treeGameState.isLose(): 
                #returns the score of game if this state was reached
                return [self.evaluationFunction(treeGameState),0]
            #This variable will hold the best action the agent can take and the score that would occur if all agents choose the optimal option
            bestValue = []
            #this value will hold the score sum of all possible actions taken by agent if they are a ghost
            expectValue = 0
            #This value holds the number of legal actions an agent can do
            legalActionNum = 0
            #Checking all legal actions
            for action in treeGameState.getLegalActions(agent):
                #Going down the tree to the next agent. if the tree goes down 1 depth, the depth variable decriments    
                if(agent == lastAgent):
                    node = expecMax(depth-1, 0, treeGameState.generateSuccessor(agent, action))
                else:
                    node = expecMax(depth, agent+1, treeGameState.generateSuccessor(agent, action))
                #if this is the first cycle of the loop, set the first action to be the best one
                if bestValue == []:
                    bestValue = node
                #if the agent is pacman and the score is higher, bestvalue is replaced with the new score and the action it takes to get there
                if(agent == 0 and node[0] >= bestValue[0]):
                    bestValue[0] = node[0]
                    bestValue[1] = action
                #if the agent is a ghost, add score to score sum
                if(agent > 0 ):
                    expectValue += node[0]
                legalActionNum += 1
            #if the agent was a ghost, set the 'best' value variable to the avereged value of all the choices
            if agent>0:
                bestValue[0] = expectValue/legalActionNum    
            return bestValue
                
        return expecMax(self.depth, self.index, gameState)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <This code used a combination of of inverted manhattan distance formulas for the food
    and power pellets in order to properly weigh the potentialy most optimal way to traverse the arena 
    without concern for the ghosts initially. A negative incentive is also placed for yet to be eaten 
    food and power pellets. The ghosts fall into the equation if the ghosts are one space away or if 
    a ghost is scared and the manhattan distance to the ghost is less than the remaining time 
    that the ghost is scared>
    """
    "*** YOUR CODE HERE ***"
    #initializing needed variables
    #pacman position
    pos = currentGameState.getPacmanPosition()
    #position of all food
    foodPos = currentGameState.getFood().asList()
    #positions of all ghosts
    newGhostStates = currentGameState.getGhostStates()
    #time left for each ghost to be scared
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #position of power pellets
    powerPellets = currentGameState.getCapsules()
    #score modifier
    scoreMod = 0
    #number of ghosts
    ghostCount = 0
    #food left for pacman to eat
    totalFood = 0
    #pellets left to eat
    totalPellets = 0
    # loop to check if a ghost fits either senario to enable their involvment
    for ghost in currentGameState.getGhostPositions():
        ghostDistance = manhattanDistance(pos, ghost)
        #old debug code
        'print(newScaredTimes)'
        'print("This is the distance to the ghost: ", manhattanDistance(newPos, ghost))'
        if ghostDistance < newScaredTimes[ghostCount]:
            scoreMod += 50
        elif ghostDistance < 2:
            scoreMod -= 1000 #discourages this idea
        ghostCount += 1
    #calculating one-forth of the inverse of the manhattan distance of each food piece for the score modification
    for food in foodPos:
        foodVal = manhattanDistance(pos, food)
        scoreMod = scoreMod + 0.25/foodVal
        totalFood += 1
    #doing the same calculation as the food for each pellet * 2
    for pellet in powerPellets:
        scoreMod = scoreMod + 0.5/manhattanDistance(pos, pellet)
        totalPellets += 1
    #discouraging uneaten food and pellets to ensure consumption
    scoreMod -= (totalFood + 3*totalPellets)
    #returns in-game score plus modifier
    return currentGameState.getScore() + scoreMod
# Abbreviation
better = betterEvaluationFunction
