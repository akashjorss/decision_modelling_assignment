"""
@author: akashmalhotra
"""

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class MRSort:
    
    def __init__(self, X, y, votes, comparison, majority_threshold):
        """constructor
        args:
            X(dataframe): feature matrix
            y(Series): labels
            votes(Series): # of votes/class"""
                
        assert(len(X.columns) == len(votes))
        self.X = X
        self.y = y
        self.votes = votes
        self.train_accuracy = np.nan
        self.comparison = comparison
        self.majority_threshold = majority_threshold
        self.initialize_thresholds()
    
    def comparator(self, a,b,criterion):
        if self.comparison[criterion]>=0:
            return a>b
        else:
            return b>a
        
    
    def initialize_thresholds(self):
        """This function initializes the threshold for X
        as average values between classes"""
        
        classes = np.sort(self.y.unique())
        pi = []
        q = 1/len(self.y.unique())
        
        for i in range(1, len(classes)):
            pi.append(self.X.quantile(i*q))

        self.pi = pd.DataFrame(pi)
        print(self.pi)
        #reverse some columns
        for col in self.X.columns:
            if self.comparison[col] > 0:
                self.pi[col] = self.pi[col][::-1].values
        self.pi.index = classes[:-1]
        
    def optimistic_classifier(self, sample, pi):
        """same as pessimistic, but optimistic"""
        for i in range(len(pi)):
            t = pi.iloc[-i-1]
            #print("threshold: ", t)
            s_pi = 0
            s_sample = 0
            for c in t.index:
                if self.comparator(sample[c], t[c], c):
                    s_sample += votes[c]
                if self.comparator(t[c], sample[c], c):
                    s_pi += votes[c]
                if (s_pi/self.votes.sum()) >= self.majority_threshold and (s_sample/self.votes.sum()) < self.majority_threshold:
                    return (len(pi)-i)
                
        return 1
            
    def pessimistic_classifier(self, sample, pi):
        """Helper function for classify:
            classify one sample into one of the classes
        args:
            sample(Series): one feature with dim (1, n_features)
        returns:
            label(int): predicted class label for this sample"""
        
        #for each threshold, check if sum of votes(normalised) is more than majority threshold
        for i in range(len(pi)): 
            t = pi.iloc[i] #for each threshold
            s = 0 #sum
            for c in t.index: #for each criterion in threshold
                if self.comparator(sample[c], t[c], c):
                    s += votes[c]
            if s/votes.sum() >= self.majority_threshold:
                return i+1 #return the class
        
        #if none of classes is returned
        return len(pi)+1
                
    
    def classify(self, X_test, pi, method):
        """classification based on thresholds for each criterion
        args:
            X_test(pd dataframe): feature matrix (n_samples, n_features)
        returns:
            y(pd series): predicted labels"""
        assert(method == 'pessimistic' or method == 'optimistic')
        y = pd.Series(np.zeros(len(X_test)), index = X_test.index)
        for i in X_test.index:
            if method == 'pessimistic':
                y[i] = self.pessimistic_classifier(X_test.loc[i], pi)
            else:
                y[i] = self.optimistic_classifier(X_test.loc[i], pi)
        
        return y


    def calculate_accuracy(self, X_test, y_test, pi, method):
        """Calculates classsification accuracy
        args:
            X_test(pd dataframe): feature matrix (n_samples, n_features)
            y_test(pd series): predicted labels"""
        y_hat = self.classify(X_test, pi, method)
        accuracy = len(y_test[y_hat == y_test])/len(y_test)
        return accuracy        

    #Find threshold
    def train(self, method, n_steps = 10, error = 0.005): 
        """This function finds threshold values for each feature based on classes
        args:
            stepsize(float): actual stepsize is (max-min)*stepsize
            num_iterations(int): obvious"""
        
        prev_max_acc = 0
        max_acc = self.calculate_accuracy(self.X, self.y, self.pi, method)
        while abs(prev_max_acc-max_acc) > error:
            prev_max_acc = max_acc
            for i in self.pi.index:
                for col in self.pi.columns:
                    if i == 1:
                        lower = self.pi.loc[2][col]
                        if self.comparison[col] == -1:
                            upper = self.X[col].min()
                        else:
                            upper = self.X[col].max()
                        #print(i, col, lower, upper)
                    elif i == len(pi):
                        upper = self.pi.loc[len(pi)-1][col]
                        if self.comparison[col] == -1:
                            lower = self.X[col].max()
                        else:
                            lower = self.X[col].min()
                    else:
                        upper = self.pi.loc[i-1][col]
                        lower = self.pi.loc[i+1][col]
                    step = abs(upper-lower)/(n_steps+1)
                    k = min(lower,upper)+step
                    pi = self.pi.copy()
                    #print(k, lower, upper)
                    while k < max(lower, upper):
                        #print(i, col, k)
                        pi.loc[i, col] = k
                        acc = self.calculate_accuracy(self.X, self.y, pi, method)
                        print(".", end="")
                        if acc > max_acc:
                            max_acc = acc
                            print('\n', i, col, "max acc : ", max_acc)
                            self.pi.loc[i,col] = k
                        k += step
                    print(i, col, max_acc)
        
        self.train_accuracy = self.calculate_accuracy(self.X, self.y, pi, method)

def load_data(filepath = "./test.csv"):
    
    data = pd.read_csv(filepath)
    
    data = data.drop(['brands', 'categories              ', 'stores',
           'countries', 'allergens', 'additives_tags', 'pnns_groups_1', 'pnns_groups_2', 
           'nutrition-score-uk_100g', 'nutrition-score-fr_100g', 'additives_n', 'nova_group'], axis = 1)
    
    #delete missing data
    data = data.dropna()
    data.index = range(len(data))
    
    #extract labels
    score_to_int = dict({'a':1, 'b':2, 'c':3, 'd':4, 'e':5})
    y = data['nutrition_grade_fr']
    for i in range(len(y)):
        y.loc[i] = score_to_int[y[i]]
    
    data = data.drop(['nutrition_grade_fr'], axis=1)

    #separate product name
    product_name = data['product_name']
    data = data.drop(['product_name'], axis=1)
    #reorder the columns
    data = data[['energy_100g', 'sugars_100g', 'saturated-fat_100g',  \
         'sodium_100g', 'proteins_100g', 'fiber_100g' ]]
    
    #type cast data as float, and labels as int
    data = data.astype(float)
    y = y.astype(int)
    
    return data, y

data, y = load_data()

#TO DO: train test split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=13)

#initialize votes
votes = pd.Series([1,1,1,1,2,2], index=data.columns)
comparison = pd.Series([-1, -1, -1, -1, 1, 1], index=data.columns)
#majority_threshold=0.5

#initialize model
l = 0.5
m = 'optimistic'
print(l, m)
mrSort = MRSort(X_train, y_train, votes, comparison, l)
mrSort.train(method=m)
mrSort.pi.to_csv(m+str(l)+"_pi.csv")

y_hat = mrSort.classify(data, mrSort.pi, m)
acc = accuracy_score(y, y_hat)
print(l, m, acc)
cm = confusion_matrix(y, y_hat, normalize="true")
pd.DataFrame(cm).to_csv(m+str(l)+"_cm.csv")

"""
pi = pd.DataFrame([
	[1550, 11, 0.8, 0.3, 10, 11],
	[1650, 14, 1, 0.4, 7, 8],
	[1750, 17, 1.7, 0.5, 4, 5],
	[1850, 20, 4, 0.6, 3, 2.5]	
    ], columns=['energy_100g', 'sugars_100g', 'saturated-fat_100g',  \
         'sodium_100g', 'proteins_100g', 'fiber_100g' ])
"""
#mrSort.majority_threshold = 0.5
y_hat = mrSort.classify(data, mrSort.pi, 'optimistic')
acc = accuracy_score(y, y_hat)
cm = confusion_matrix(y, y_hat, normalize="true")

print(acc)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        print(round(cm[i][j],2), end=' | ')
    print('\n------------------------------------')

print('\n')
for i in range(len(mrSort.pi)):
    for j in mrSort.pi.columns:
        print(round(mrSort.pi.iloc[i][j],2), end=' | ')
    print('\n------------------------------------')




