import numpy as np
from keras.datasets import mnist

def create_subset(n = 1000):
    """ 
    Create subset of MNIST dataset of length n, default 1000
    """
    (trainX, trainY), (testX, testY) = mnist.load_data()

    trainX_subset = trainX[:n]
    trainY_subset = trainY[:n]
    testX_subset = testX[:n]
    testY_subset = testY[:n]
    
    return trainX_subset, trainY_subset, testX_subset, testY_subset

def save_file(trainX_subset, trainY_subset, testX_subset, testY_subset, name = "subset"):
    np.savez_compressed(f'{name}.npz', x_train=trainX_subset, y_train=trainY_subset, x_test=testX_subset, y_test=testY_subset)

def receive_subset_info():
    print("Input a number from 1-60,000 for the size of the MNIST subset:")
    while True:
        try:
            n = int(input())
            if n >= 1 and n <= 60000:
                break
        except:
            pass
        print("Invalid input try again:")
    
    
    trainX_subset, trainY_subset, testX_subset, testY_subset = create_subset(n)
    save_file(trainX_subset, trainY_subset, testX_subset, testY_subset, "config")

