# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from asyncio.windows_events import NULL
from telnetlib import theNULL
from xml.dom.minicompat import NodeList
from game import Directions
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    from util import Stack
    #stacks for the path and place
    pathStack = Stack()
    placeStack = Stack()
    #areas that have already been visited
    visited = []
    #setting up initial start location
    placeStack.push(problem.getStartState())
    pathStack.push([])
    #looping until a path or no path is found
    while 1:
        #if the stack is empty, that means that there is no good path
        if placeStack.isEmpty():
            return[]
        #popping off top of the stacks to check
        place = placeStack.pop()
        path = pathStack.pop()
        #has this node been visited yet? skip if so, execute if not
        if place not in visited:
            visited.append(place)
            #goal has been found! Return the path
            if problem.isGoalState(place):
                return path
            #adding all possible routes in unvisited areas in case one route comes back unsucessful
            nextstates = problem.getSuccessors(place)
        
            for state in (nextstates):
                #print(state)
                placeStack.push(state[0])
                pathStack.push(path + [state[1]]) 
               
#man I really miss brackets    
    


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    #initialize variables
    PathToReturn = []
    path = Queue()
    visited = []
    queue = Queue() #queue for unvisited children
    
    #start case
    root = problem.getStartState() #secure the root
    visited.append(root) #add root to visited nodes
    #print(root, "was added to visited.")
    queue.push(root)
    path.push([])

    while not queue.isEmpty(): #while the queue is not empty, continue exploring

        #get current node and path
        curr = queue.pop()
        currpath = path.pop()
 
        if problem.isGoalState(curr):
           #print("Found it!") #this is the goal state
           PathToReturn = currpath
           break

        for x in problem.getSuccessors(curr):
            if(x[0] in visited): #root is already in visited, and is the first case, so its children will be added
                #print("This node was not added to queue.", x[0])
                continue #do not put this in queue
            visited.append(x[0])
            #print(x[0], "was added to visited.")
            path.push(currpath + [x[1]]) #appends the path
            queue.push(x[0]) #adds children info to queue   
    
    #print("this is the length of the return path.", (len(PathToReturn)))
    #print(PathToReturn)
    #print("These are the nodes that BFS visited:", visited)
    return PathToReturn

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    #priority queue for path, place, and cost
    pathPlaceCost = PriorityQueue()
    #set for visited nodes
    visited = set()
    #push initial state into queue 
    pathPlaceCost.push([[],problem.getStartState(), 0],0)

    while True:
        #is queue empty? than no solution!
        if pathPlaceCost.isEmpty():
            return []
        #pop next item off of queue
        path,place,cost = pathPlaceCost.pop()
        #if place has already been visited, skip it!
        if place not in visited:
            visited.add(place)
            #is this the goal? Return path if so
            if problem.isGoalState(place):
                return path
            #push all successors into priority queue with new costs
            nextstates = problem.getSuccessors(place)
            for state in nextstates:
                pathPlaceCost.push([path + [state[1]], state[0], state[2] + cost], state[2] + cost)
                

    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    #priority queue for path, place, and cost
    pathPlaceCost = PriorityQueue()
    #set for visited nodes
    visited = set()
    #push initial state into queue
    pathPlaceCost.push([[],problem.getStartState(), 0],0)
    while True:
        #is queue empty? than no solution!
        if pathPlaceCost.isEmpty():
            return []
        #pop next item off of queue
        path,place,cost = pathPlaceCost.pop()
        #if place has already been visited, skip it!
        if tuple(place) not in visited:
            visited.add(tuple(place))
            #is this the goal? Return path if so
            if problem.isGoalState(place):
                return path
            #push all successors into priority queue with new costs
            nextstates = problem.getSuccessors(place)
            for state in nextstates:
                pathPlaceCost.push([path + [state[1]], state[0], state[2] + cost], state[2] + cost + heuristic(state[0], problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
