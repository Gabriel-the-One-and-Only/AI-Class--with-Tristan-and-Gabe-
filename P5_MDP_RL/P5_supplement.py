# P5_supplement.py
# -----------
# Reference: this code base is adapted from ai.berkeley.edu, see the license information as follows:
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



######################
# Student Information #
######################
## Write your name and your partner name here
#Gabriel Shultz and Tristan Jibben


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0 #if there is no chance it will fall off the bridge, it will try to exit the map as quickly as possible.
    #otherwise it thinks the best thing to do is not attempt. Changing the discount did not help it try to cross.
    #Any negative disount made it head to the bad region which makes sense. With positive values, the policy was to move back to origin from the first
    #two squares to the left of start. The middle would either head to start or the red tile above it. And the two squares closest to the end went to the end.
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = 0.2
    answerNoise = 0.01
    answerLivingReward = -0.001
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = 0.6
    answerNoise = 0.01
    answerLivingReward = -0.2
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = 0.9
    answerNoise = 0.1
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = 20
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE'
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import P5_supplement
    for q in [q for q in dir(P5_supplement) if q.startswith('question')]:
        response = getattr(P5_supplement, q)()
        print('  Question %s:\t%s' % (q, str(response)))
