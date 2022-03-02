# shopSmart.py
# ------------
# This codebase is adapted from UC Berkeley AI. Please see the following information about the license.
# This codebase is adapted from UC Berkeley AI. Please see the following information about the license.
# Licensing Information from the :  You are free to use or extend these projects for
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
Here's the intended output of this script, once you fill it in:

Welcome to grocery shop 1
Welcome to grocery shop 2
For orders:  [('apples', 1.0), ('oranges', 3.0)] best shop is shop1
For orders:  [('apples', 3.0)] best shop is shop2
"""
from __future__ import print_function
import shop


def shopSmart(orderList, groceryShops):
    """
        orderList: List of (groceryItems, numPound) tuples
        groceryShops: List of grocery Shops
    """
    "*** YOUR CODE HERE ***"
    lowestCostShop = groceryShops[0]
    for x in groceryShops:
        if lowestCostShop.getPriceOfOrder(orderList) > x.getPriceOfOrder(orderList):
            lowestCostShop = x
  
    return lowestCostShop


if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orders = [('apples', 1.0), ('oranges', 3.0)]
    cost1 = {'apples': 2.0, 'oranges': 1.0}
    shop1 = shop.groceryShop('shop1', cost1)
    cost2 = {'apples': 1.0, 'oranges': 5.0}
    shop2 = shop.groceryShop('shop2', cost2)
    shops = [shop1, shop2]
    print(shopSmart(orders, shops))
    print("For orders ", orders, ", the best shop is", shopSmart(orders, shops).getName())
    orders = [('apples', 3.0)]
    print("For orders: ", orders, ", the best shop is", shopSmart(orders, shops).getName())
