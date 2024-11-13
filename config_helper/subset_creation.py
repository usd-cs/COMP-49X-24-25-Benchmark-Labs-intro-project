import numpy as np
from keras.datasets import mnist

def create_subset(n = 1000, name = "subset"):
    """ 
    Create subset of MNIST dataset of length n, default 1000
    """
    (trainX, trainY), (testX, testY) = mnist.load_data()

    trainX_subset = trainX[:n]
    trainY_subset = trainY[:n]
    testX_subset = testX[:n]
    testY_subset = testY[:n]

    np.savez_compressed(f'{name}.npz', x_train=trainX_subset, y_train=trainY_subset, x_test=testX_subset, y_test=testY_subset)
    
    return trainX_subset, trainY_subset, testX_subset, testY_subset

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
    
    print("Now enter a name for the file to be saved:")
    while True: 
        try:
            name = str(input())
            break
        except:
            pass
        print("Invalid input try again:")
    
    create_subset(n, name)

receive_subset_info()

def test_subset():
    trainX, trainY, testX, testY = create_subset(100)
    # .shape outputs the size, which should be 100 and the pixel x pixel of the image, which should stay as 28 x 28
    # X values refer to images and Y values refer to labels which should be just 100 labels, 1 for each image in the dataset
    assert trainX.shape == (100,28,28)
    assert trainY.shape == (100,)
    assert testX.shape == (100,28,28)
    assert testY.shape == (100,)