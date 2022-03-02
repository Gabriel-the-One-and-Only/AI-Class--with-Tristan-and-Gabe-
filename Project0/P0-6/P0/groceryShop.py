# groceryShop.py
# -----------------
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
To run this script, type

  python groceryShop.py

Once you have correctly implemented the buyLotsOfFruit function,
the script should produce the output:

Cost of [('apples', 4.0), ('carrot', 2.0), ('celery', 3.0)] is 18.1
"""
from __future__ import print_function

groceryPrices = {'apples': 2.50, 'oranges': 1.0, 'pears': 3.5, 'celery': 2.1, 'carrot':1.30,
'limes': 0.75, 'strawberries': 2.50}
### groceryPrices is the list of prices per pound

def groceryShop(orderList):

    totalCost = 0.0
    for x in orderList:
        print(x)
        cost = groceryPrices[x[0]]*x[1]
        totalCost += cost
        if(cost == 0):
            print("This item is not in the list of prices")
        
    return totalCost


# Main Method
if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orderList = [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)]
    print('Cost of', orderList, 'is', groceryShop(orderList))
