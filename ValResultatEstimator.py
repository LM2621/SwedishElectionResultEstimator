from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

# Preprocessing of data:
# All data is collected from SCB (Statistiska central byrån)
# Some feature engineering in the proprocessing ( for example calculating percentage from the total,
# calculating most voted party per municipality, etc)
# Normalization of columns, like average salary, has been tried but did not improve score
# Currently best performance of this model is 0.75

valData = pd.read_csv("/home/max/Documents/Data science projects/ValEstimator/MasterSheet.csv")
valData.drop(['Kommun'], axis=1, inplace=True)

# Create an instance of LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# Label encode the 'category' column
valData['partiEncoded'] = encoder.fit_transform(valData['Mestrostadeparti'])
targetVar = valData.partiEncoded
valData.drop(['Mestrostadeparti'], axis=1, inplace=True)

# Compute the correlation matrix
corr_matrix = valData.corr()

# Create a correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

valData.drop(['partiEncoded'], axis=1, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(valData,
                                                    targetVar,
                                                    test_size=1 / 20)
#Import cross validation scorer
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
predictions = rf_classifier.predict(X_test)

cv = cross_val_score(rf_classifier, X_train, y_train, cv=3)
print("random forest score:")
print(cv)
print(cv.mean())

# Create an instance of the logistic regression classifier
logreg = LogisticRegression(multi_class='ovr', solver='liblinear')

# Tune Log Reg hyperparameters by using gridsearch
from sklearn.model_selection import GridSearchCV

param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'max_iter': [100, 500, 1000],
    'tol': [1e-4, 1e-3, 1e-2],
    'class_weight': [None, 'balanced'],
    'random_state': [None, 42, 2021]
}

gridsearch = GridSearchCV(logreg, param_grid = param_grid, cv = 3, verbose = False, n_jobs = -1)
gridsearchedLogReg = gridsearch.fit(X_train, y_train)

cv = cross_val_score(gridsearchedLogReg, X_train, y_train, cv=3)
print("Grid searched Logistic regression score:")
print(cv)
print(cv.mean())

# Fit the model to the training data
logreg.fit(X_train, y_train)

cv = cross_val_score(logreg, X_train, y_train, cv=3)
print("Logistic regression score:")
print(cv)
print(cv.mean())

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc.fit(X_train, y_train)

cv = cross_val_score(gbc, X_train, y_train, cv=3)
print("Gradient Booster score:")
print(cv)
print(cv.mean())

from sklearn.naive_bayes import MultinomialNB
# Create and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

cv = cross_val_score(nb_classifier, X_train, y_train, cv=3)
print("Naive Bayes score:")
print(cv)
print(cv.mean())

print("Done")