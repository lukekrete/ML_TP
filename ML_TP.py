import pandas
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
from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy

def main():
    test = [[3,2,1320,5300,2,0,0,4,6,550,700,1997,2010,98179,47.632,-122.426,1580,5300]]
    features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement','yr_built','yr_reno',
            'zipcode','lat','long','sqft_living15','sqft_lot15']
    data = pandas.read_csv('kc_house_data.csv',names=features)
    d = pandas.DataFrame(data,columns=features[1:])
    t = pandas.DataFrame(data,columns=[features[0]])
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
    
    
    
