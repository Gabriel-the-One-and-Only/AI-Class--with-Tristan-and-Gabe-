### code base: ai.berkeley.edu

import nn


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #size of the hidden layers
        layerSize = 100
        #size of each batch
        self.batchSize = 50
        #'w' stands for weight while 'b' stands for the bias
        #hidden layer 1
        self.w1 = nn.Parameter(784,layerSize)
        self.b1 = nn.Parameter(1,layerSize)
        #hidden layer 2
        self.w2 = nn.Parameter(layerSize,layerSize)
        self.b2 = nn.Parameter(1,layerSize)
        #output layer
        self.wOut = nn.Parameter(layerSize, 10)
        self.bOut = nn.Parameter(1,10)
        

    def run(self, x):
        """
         f(x)=relu(x⋅W1+b1)⋅W2+b is the funciton to implement
         
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #crunching the first layer
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x,self.w1),self.b1))
        #crunching the second layer with the first layer logits as an input
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1,self.w2),self.b2))
        #crunshing and returning the output logits using the second layer logits as an input
        return nn.AddBias(nn.Linear(layer2,self.wOut),self.bOut)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        #returns the loss by running the nural net and comparing the calculated labels to the correct ones
        return nn.SoftmaxLoss(self.run(x),y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #starting learn rate, changes later on
        learnRate = 0.05
        #index variable for later use
        index = 0
        #for the given image data and correct label to that data in new batch set
        for x,y in dataset.iterate_forever(self.batchSize):
            #gradient slope as a vector
            slope = nn.gradients(self.get_loss(x,y),[self.w1,self.b1,self.w2,self.b2,self.wOut,self.bOut])\
            #update weights and biases of layer 1
            self.w1.update(slope[0],-learnRate)
            self.b1.update(slope[1],-learnRate)
            #update weights and biases of layer 2
            self.w2.update(slope[2],-learnRate)
            self.b2.update(slope[3],-learnRate)
            #update weights and biases of output layer
            self.wOut.update(slope[4],-learnRate)
            self.bOut.update(slope[5],-learnRate)
            #if ~6-7 epochs have been computed, lower the learning rate to 0.005 and check for accuracy  
            if index > 7500:
                learnRate = 0.005
                #only checked later due to how computationaly intensive this is
                if dataset.get_validation_accuracy() > 0.972:
                    return
            else:
                index += 1
        
        #create batches from the dataset
        
        #need to run the algorith for each element in the batch
        '''
        oldParameters = [self.w1, self.b1, self.w2, self.b2]
        newParameters = nn.Linear(nn.Linear(nn.gradients(self.get_loss, oldParameters ), oldParameters), 0.01) #arbitrary step size (learning rate?)
        #for i in range(len(newParameters)):
        self.w1 = newParameters[0]
        self.b1 = newParameters[1]
        self.w2 = newParameters[2]
        self.b2 = newParameters[3]
        '''
            


