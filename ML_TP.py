import pandas
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import neighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy
import matplotlib.pyplot as plt

def main():
    test = [[3,2,1320,5300,2,0,0,4,6,550,700,1997,2010,98179,47.632,-122.426,1580,5300]]
    features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement','yr_built','yr_reno',
            'zipcode','lat','long','sqft_living15','sqft_lot15']
    data = pandas.read_csv('kc_house_data.csv',names=features)
    test_data = pandas.DataFrame(data,columns=features)
    d = pandas.DataFrame(data,columns=features[1:])
    d.plot()
    plt.show()
    t = pandas.DataFrame(data,columns=[features[0]])
    #-----------GRAPHING-----------------------------------
    plt.scatter('sqft_living','price',data=test_data)
    plt.xlabel("Square Footage Living Space")
    plt.ylabel("Price of House")
    plt.title("Square feet of living space vs price of house")
    plt.show()
    '''
    plt.plot('sqft_living','price',data=test_data)
    plt.xlabel("Square Footage Living Space")
    plt.ylabel("Price of House")
    plt.title("Square feet of living space vs price of house")
    plt.show()
    '''
    plt.scatter('bedrooms','price',data=test_data)
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Price of house")
    plt.title("Number of Bedrooms vs price of house")
    plt.show()
    #TEST
    #for i in range(len(features)-1):
    #    plt.scatter(features[i+1],features[0],data=test_data)
    #    plt.xlabel(features[i+1])
     #   plt.ylabel(features[0])
      #  plt.show()
    fig, axes = plt.subplots(ncols=5, nrows=4, figsize=(20,15))
    #for i, ax in enumerate(axes.flatten()):
    #    if i < 19:
    #        test_data[test_data.columns[i]].plot(color='green',ax=ax)
    #        ax.set_title(test_data.columns[i])
    #plt.tight_layout()

    i = 0
    axes = axes.flatten()
    for k,v in test_data.items():
        sns.distplot(v,ax=axes[i])
        i+=1
    plt.tight_layout()
    plt.show()
    print(d.describe())
    print()
    print("---------------------------------------------------")
    print("Test house to use with K-Neighbors Classifier:")
    for i in range(18):
        print(str(features[i+1]) + ": " + str(test[0][i]))
    print("---------------------------------------------------")
    print("Testing K-NN Classifier with the test house")
    knn = neighbors.KNeighborsClassifier(n_neighbors=3, metric='euclidean').fit(d,np.ravel(t))
    print("Test with k = 3 and Euclidean Distance")
    print(knn.predict(test))
    
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, metric='euclidean').fit(d,np.ravel(t))
    print("Test with k = 5 and Euclidean Distance")
    print(knn.predict(test))

    knn = neighbors.KNeighborsClassifier(n_neighbors=7, metric='euclidean').fit(d,np.ravel(t))
    print("Test with k = 7 and Euclidean Distance")
    print(knn.predict(test))

    knn = neighbors.KNeighborsClassifier(n_neighbors=27, metric='euclidean').fit(d,np.ravel(t))
    print("Test with k = 27 and Euclidean Distance")
    print(knn.predict(test))

    
    




    xtrain,xtest,ytrain,ytest = train_test_split(d,t,test_size=0.3)
    lr = LinearRegression()
    lr = lr.fit(xtrain,ytrain)
    prediction = lr.predict(xtest)
    r = lr.score(xtest,ytest)
    print("---------------------------------------------------")
    print("Testing Linear Regression with a Train/Test split of 70%/30%")
    print("Accuracy of Linear Regression when normalize = False: ",end='')
    print(r)
    #--AVG--
    count = 0
    for i in range(len(prediction)):
        count = count + prediction[i]
    count = count/len(prediction)
    print("Average value of a house from the predicted data = " + str(count))
    print()
    #-------
    lr = LinearRegression(normalize=True)
    lr = lr.fit(xtrain,ytrain)
    prediction = lr.predict(xtest)
    r = lr.score(xtest,ytest)
    print("Accuracy of Linear Regression when noramlize = True: ",end='')
    print(r)
    #--AVG--
    count = 0
    for i in range(len(prediction)):
        count = count + prediction[i]
    count = count/len(prediction)
    print("Average value of a house from the predicted data = " + str(count))
    #-------
    print()
    
    print("---------------------------------------------------")
    print("Testing A Random Forest Regressor with a Train/Test spit of 70%/30%")
    rfr = RandomForestRegressor(n_estimators=10)
    rfr = rfr.fit(xtrain,np.ravel(ytrain))
    prediction = rfr.predict(xtest)
    r = rfr.score(xtest,ytest)
    print("Accuracy of Random Forest Regressor when n_estimators = 10: ",end='')
    print(r)
    #--AVG--
    count = 0
    for i in range(len(prediction)):
        count = count + prediction[i]
    count = count/len(prediction)
    print("Average value of a house from the predicted data = " + str(count))
    #-------
    print()
    rfr = RandomForestRegressor(n_estimators=30)
    rfr = rfr.fit(xtrain,np.ravel(ytrain))
    prediction = rfr.predict(xtest)
    r = rfr.score(xtest,ytest)
    print("Accuracy of Random Forest Regressor when n_estimators = 30: ",end='')
    print(r)
    #--AVG--
    count = 0
    for i in range(len(prediction)):
        count = count + prediction[i]
    count = count/len(prediction)
    print("Average value of a house from the predicted data = " + str(count))
    #-------
    print()
    rfr = RandomForestRegressor(n_estimators=60)
    rfr = rfr.fit(xtrain,np.ravel(ytrain))
    prediction = rfr.predict(xtest)
    r = rfr.score(xtest,ytest)
    print("Accuracy of Random Forest Regressor when n_estimators = 60: ",end='')
    print(r)
    #--AVG--
    count = 0
    for i in range(len(prediction)):
        count = count + prediction[i]
    count = count/len(prediction)
    print("Average value of a house from the predicted data = " + str(count))
    #-------
    print()
    print("---------------------------------------------------")
    print("Testing K-Neighbors Regressor with a Train/Test spit of 70%/30%")
    knr = KNeighborsRegressor(n_neighbors=3)
    knr = knr.fit(xtrain,np.ravel(ytrain))
    prediction = knr.predict(xtest)
    r = knr.score(xtest,ytest)
    print("Accuracy of K-Neighbors Regressor when k = 3: ",end='')
    print(r)
    #--AVG--
    count = 0
    for i in range(len(prediction)):
        count = count + prediction[i]
    count = count/len(prediction)
    print("Average value of a house from the predicted data = " + str(count))
    #-------
    print()
    knr = KNeighborsRegressor(n_neighbors=5)
    knr = knr.fit(xtrain,np.ravel(ytrain))
    prediction = knr.predict(xtest)
    r = knr.score(xtest,ytest)
    print("Accuracy of K-Neighbors Regressor when k = 5: ",end='')
    print(r)
    #--AVG--
    count = 0
    for i in range(len(prediction)):
        count = count + prediction[i]
    count = count/len(prediction)
    print("Average value of a house from the predicted data = " + str(count))
    #-------
    print()
    knr = KNeighborsRegressor(n_neighbors=10)
    knr = knr.fit(xtrain,np.ravel(ytrain))
    prediction = knr.predict(xtest)
    r = knr.score(xtest,ytest)
    print("Accuracy of K-Neighbors Regressor when k = 10: ",end='')
    print(r)
    #--AVG--
    count = 0
    for i in range(len(prediction)):
        count = count + prediction[i]
    count = count/len(prediction)
    print("Average value of a house from the predicted data = " + str(count))
    #-------
    print()
