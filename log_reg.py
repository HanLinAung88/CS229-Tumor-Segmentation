from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import util
from glob import glob
from matplotlib.pyplot import imread, imshow
import matplotlib.pyplot as plt
import cv2
from sklearn import metrics

if __name__ == "__main__":
    #extracts data
    X, Y = util.extract_data_CBIS_MIAS(isCBIS=True, isMias=False)
    print(X.shape)
    print(Y.shape)

    print(X[0].shape)
    print(Y[0].shape)

    print(X[1].shape)
    print(Y[1].shape)
    plt.imshow(X[0])
    plt.show()

    plt.imshow(Y[0])
    plt.show()

    plt.imshow(X[1])
    plt.show()

    plt.imshow(Y[1])
    plt.show()
    X_reg, Y_reg = util.extract_logReg_data(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg,train_size=0.6,random_state=42)

    #initializes classifier
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, Y_train)

    #prediction
    predictions = clf.predict(X_test)
    print(predictions)

    #accuracy metrics
    accuracy = np.mean(predictions == Y_test)
    print(Y_test.shape)
    print(predictions.shape)
    print(metrics.classification_report(Y_test, predictions))
    print('Accuracy: ', accuracy)
