# multiAgents.py
# --------------
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


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        succVal = (successorGameState.getScore() - currentGameState.getScore()) / 5

        minFoodDist = float('inf')
        for food in newFood.asList():
            foodDist = abs(food[0] - newPos[0]) + abs(food[1] - newPos[1])
            if foodDist < minFoodDist:
                minFoodDist = foodDist
        foodVal = 1/minFoodDist * 1.5

        minGhostDist = float('inf')
        for ghost in newGhostStates:
            if abs(newPos[0] - ghost.getPosition()[0]) + abs(newPos[1] - ghost.getPosition()[1]) <= minGhostDist:
                minGhostDist = abs(newPos[0] - ghost.getPosition()[0]) + abs(newPos[1] - ghost.getPosition()[1])
        posVal = -1/minGhostDist if minGhostDist > 0 else -100

        return succVal + posVal + foodVal


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"
        maxChild = -float('inf')
        for action in gameState.getLegalActions(0):
            child = gameState.generateSuccessor(0, action)
            if self.computeValue(child, self.depth, 1) > maxChild:
                maxChild = self.computeValue(child, self.depth, 1)
                act = action
        return act

    def computeValue(self, gameState, d, agent):
        if gameState.isLose() or gameState.isWin():
            return gameState.getScore()
        elif d == 0:
            return self.evaluationFunction(gameState)

        if agent == 0:
            value = -float('inf')
            for action in gameState.getLegalActions(agent):
                child = gameState.generateSuccessor(agent, action)
                value = max(value, self.computeValue(child, d, agent+1))
        elif agent == gameState.getNumAgents()-1:
            value = float('inf')
            for action in gameState.getLegalActions(agent):
                child = gameState.generateSuccessor(agent, action)
                value = min(value, self.computeValue(child, d-1, 0))
        else:
            value = float('inf')
            for action in gameState.getLegalActions(agent):
                child = gameState.generateSuccessor(agent, action)
                value = min(value, self.computeValue(child, d, agent+1))
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxChild = -float('inf')
        alpha = -float('inf')
        for action in gameState.getLegalActions(0):
            child = gameState.generateSuccessor(0, action)
            if self.computeValue(child, self.depth, 1, alpha, float('inf')) > maxChild:
                maxChild = self.computeValue(child, self.depth, 1, alpha, float('inf'))
                alpha = maxChild
                act = action
        return act

    def computeValue(self, gameState, d, agent, alpha, beta):
        if gameState.isLose() or gameState.isWin():
            return gameState.getScore()
        elif d == 0:
            return self.evaluationFunction(gameState)

        if agent == 0:
            for action in gameState.getLegalActions(agent):
                child = gameState.generateSuccessor(agent, action)
                alpha = max(alpha, self.computeValue(child, d, agent+1, alpha, beta))
                if beta <= alpha:
                    break
            return alpha
        elif agent == gameState.getNumAgents()-1:
            for action in gameState.getLegalActions(agent):
                child = gameState.generateSuccessor(agent, action)
                beta = min(beta, self.computeValue(child, d-1, 0, alpha, beta))
                if beta <= alpha:
                    break
            return beta
        else:
            for action in gameState.getLegalActions(agent):
                child = gameState.generateSuccessor(agent, action)
                beta = min(beta, self.computeValue(child, d, agent+1, alpha, beta))
                if beta <= alpha:
                    break
            return beta


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
        maxChild = -float('inf')
        for action in gameState.getLegalActions(0):
            child = gameState.generateSuccessor(0, action)
            if self.computeValue(child, self.depth, 1) > maxChild:
                maxChild = self.computeValue(child, self.depth, 1)
                act = action
        return act

    def computeValue(self, gameState, d, agent):
        if gameState.isLose() or gameState.isWin():
            return gameState.getScore()
        elif d == 0:
            return self.evaluationFunction(gameState)

        if agent == 0:
            value = -float('inf')
            for action in gameState.getLegalActions(agent):
                child = gameState.generateSuccessor(agent, action)
                value = max(value, self.computeValue(child, d, agent+1))
        elif agent == gameState.getNumAgents()-1:
            value = 0
            for action in gameState.getLegalActions(agent):
                child = gameState.generateSuccessor(agent, action)
                value += self.computeValue(child, d-1, 0)
            value /= len(gameState.getLegalActions(agent))
        else:
            value = 0
            for action in gameState.getLegalActions(agent):
                child = gameState.generateSuccessor(agent, action)
                value += self.computeValue(child, d, agent+1)
            value /= len(gameState.getLegalActions(agent))
        return value


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This function computes the value of a state by adding or subtracting the distance from the nearest
      ghost, the distance to the nearest food, the number of food on the map, and that state's score. Each of these
      values is multiplied by some constant depending on how valuable they are.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        value = -float('inf')
    elif currentGameState.isWin():
        value = float('inf')
    else:
        score = currentGameState.getScore()
        pos = currentGameState.getPacmanPosition()
        food = currentGameState.getFood()
        ghostStates = currentGameState.getGhostStates()
        scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

        minFoodDist = float('inf')
        for f in food.asList():
            foodDist = abs(f[0] - pos[0]) + abs(f[1] - pos[1])
            if foodDist < minFoodDist:
                minFoodDist = foodDist

        minGhostDist = float('inf')
        for ghost in ghostStates:
            ghostDist = abs(pos[0] - ghost.getPosition()[0]) + abs(pos[1] - ghost.getPosition()[1])
            if ghostDist < minGhostDist:
                minGhostDist = ghostDist

        numFood = currentGameState.getNumFood()

        value = minGhostDist*2 - minFoodDist - numFood*5 + score

    return value


# Abbreviation
better = betterEvaluationFunction
