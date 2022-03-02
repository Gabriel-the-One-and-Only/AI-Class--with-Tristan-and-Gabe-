# shop.py
# -------
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


class groceryShop:

    def __init__(self, name, groceryPrices):
        """
            name: Name of the groceryShop

            groceryPrices: Dictionary with keys as grocery item
            strings and prices for values e.g.
            {'apples':2.00, 'oranges': 1.50, 'strawberries': 2.75}
        """
        self.groceryPrices = groceryPrices
        self.name = name
        print('Welcome to %s grocery shop' % (name))

    def getCostPerPound(self, groceryItem):
        """
            groceryItem: groceryItem string
        Returns cost of 'groceryItem', assuming 'groceryItem'
        Exists at that store or None otherwise
        """
        if groceryItem not in self.groceryPrices:
            return None
        return self.groceryPrices[groceryItem]

    def getPriceOfOrder(self, orderList):
        """
            orderList: List of (groceryItem, numPounds) tuples

        Returns cost of orderList, only including the values of
        grocery items that this groceryshop has.
        """
        totalCost = 0.0
        for groceryItem, numPounds in orderList:
            costPerPound = self.getCostPerPound(groceryItem)
            if costPerPound != None:
                totalCost += numPounds * costPerPound
        return totalCost

    def getName(self):
        return self.name

    def __str__(self):
        return "<groceryShop: %s>" % self.getName()

    def __repr__(self):
        return str(self)
