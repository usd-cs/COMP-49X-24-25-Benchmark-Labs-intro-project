from subset_creation import create_subset
import pytest

def test_create_subset():
    trainX, trainY, testX, testY = create_subset(100)
    # .shape outputs the size, which should be 100 and the pixel x pixel of the image, which should stay as 28 x 28
    # X values refer to images and Y values refer to labels which should be just 100 labels, 1 for each image in the dataset
    assert trainX.shape == (100,28,28)
    assert trainY.shape == (100,)
    assert testX.shape == (100,28,28)
    assert testY.shape == (100,)