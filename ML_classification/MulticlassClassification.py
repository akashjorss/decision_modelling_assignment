import os

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd




def plot_result(clf, X_test, y_test, model):
    dirName = 'result'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    plot = plot_confusion_matrix(clf, X_test, y_test, display_labels=['A', 'B', 'C', 'D', 'E'], cmap=plt.cm.Blues)
    plot.ax_.set_title("Confusion Matrix of " + model)
    plt.savefig("result/"+model + "_result.png")


class Classification_train:
    def __init__(self, path, test_size, model):
        """
        path: data set path
        test_size: from 0-1, the percentage of test set
        model: random_forest/svc
        """
        self.path = path
        self.test_size = test_size
        self.model = model

    def data_preparation(self):
        data = pd.read_csv(self.path, encoding='utf-8')
        # split training and testing set
        labelencoder_y = LabelEncoder()
        feature = data[
            ['energy_100g', 'saturated-fat_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'sodium_100g']]
        target = labelencoder_y.fit_transform(data['nutrition_grade_fr'])

        X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=self.test_size, random_state=45)

        # Feature scaling: Make sure the features on th same scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train_model(self):
        if self.model == "random_forest":
            clf, X_test, y_test = self.randomForest_train()
            plot_result(clf, X_test, y_test, self.model)
        elif self.model == "svc":
            clf, X_test, y_test = self.svc_train()
            plot_result(clf, X_test, y_test, self.model)
        return clf

    def randomForest_train(self):
        X_train, X_test, y_train, y_test = self.data_preparation()
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("predict test_set label with random forest\n", classification_report(y_test, predictions))
        return clf, X_test, y_test

    def svc_train(self):
        X_train, X_test, y_train, y_test = self.data_preparation()
        clf = OneVsRestClassifier(
            SVC(kernel='rbf', class_weight="balanced"))  # choose between different kernel trick, rbf perform better
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("predict test_set label with svc\n", classification_report(y_test, predictions))
        return clf, X_test, y_test


class Predict:
    def __init__(self, path, model, clf):
        self.path = path
        self.model = model
        self.clf = clf

    def predict(self):
        data = pd.read_csv(self.path, encoding='utf-8')
        model = self.model
        clf = self.clf
        # split training and testing set
        labelencoder_y = LabelEncoder()
        X_test = data[['energy_100g', 'saturated-fat_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'sodium_100g']]
        y_test = labelencoder_y.fit_transform(data['nutrition_grade_fr'])

        # Feature scaling: Make sure the features on th same scale
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)

        predictions = clf.predict(X_test)

        print("predict label for new dataset with"+model+"\n", classification_report(y_test, predictions))
        if model == "random_forest":
            model = "predict_random_forest"
        elif model == "svc":
            model = "predict_svc"
        plot_result(clf, X_test, y_test, model)


# train random forest model by openfoodfacts_simplified_database_clean.csv then use the model to predict the OpenFood_Petales.csv
classification_random_forest = Classification_train("dataset/openfoodfacts_simplified_database_clean.csv", 0.2, "random_forest")
classification_random_forest.data_preparation()
clf_rf = classification_random_forest.train_model()
predict_random_forest = Predict("dataset/OpenFood_Petales.csv", "random_forest", clf_rf)
predict_random_forest.predict()
#
# train svc model by openfoodfacts_simplified_database_clean.csv then use the model to predict the OpenFood_Petales.csv
classification_random_forest = Classification_train("dataset/openfoodfacts_simplified_database_clean.csv", 0.2, "svc")
classification_random_forest.data_preparation()
clf_svc = classification_random_forest.train_model()
predict_random_forest = Predict("dataset/OpenFood_Petales.csv", "svc", clf_svc)
predict_random_forest.predict()
