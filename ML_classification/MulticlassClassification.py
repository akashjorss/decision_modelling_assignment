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


class Classification:
    def __init__(self, path):
        self.path = path

    def data_preparation(self):
        data = pd.read_csv(self.path, encoding='utf-8')
        # split training and testing set
        labelencoder_y = LabelEncoder()
        feature = data[
            ['energy_100g', 'saturated-fat_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'sodium_100g']]
        target = labelencoder_y.fit_transform(data['nutrition_grade_fr'])

        X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=45)

        # Feature scaling: Make sure the features on th same scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def randomForest(self):
        X_train, X_test, y_train, y_test = self.data_preparation()
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        f1 = f1_score(y_test, predictions, average='micro')
        print(classification_report(y_test, predictions))
        plot = plot_confusion_matrix(clf, X_test, y_test, display_labels=['A', 'B', 'C', 'D', 'E'], cmap=plt.cm.Blues)
        plot.ax_.set_title("Confusion Matrix of Random Forest Performance")
        plt.savefig("RandomForest_result")

    def svm(self):
        X_train, X_test, y_train, y_test = self.data_preparation()
        clf = OneVsRestClassifier(
            SVC(kernel='rbf', class_weight="balanced"))  # choose between different kernel trick, rbf perform better
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print(classification_report(y_test, predictions))
        plot = plot_confusion_matrix(clf, X_test, y_test, display_labels=['A', 'B', 'C', 'D', 'E'], cmap=plt.cm.Blues)
        plot.ax_.set_title("Confusion Matrix of rbf SVM")
        plt.savefig("SVM_result")


classification = Classification("dataset/openfoodfacts_simplified_database_clean.csv")
classification.randomForest()
classification.svm()
