# Import necessary modules for logistic regression and splitting dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# All data is collected from SCB (Statistiska central byrån)
# Some feature engineering in the proprocessing ( for example calculating percentage from the total,
# calculating most voted party per municipality, etc)
# Normalization of columns, like average salary, has been tried but did not improve score
# Plots the results as an interactive map, showing the municipality, truth and estimate
# Currently best performance of this model is 0.75

# Read CSV file into DataFrame
valData = pd.read_csv("/home/max/Documents/Data science projects/ValEstimator/MasterSheetWithMunicipalityCoordinates.csv")
# Create a copy of the DataFrame
valDataMedKommun = valData.copy()
# Drop 'Kommun' and 'Koordinater' columns from the DataFrame
valData.drop(['Kommun'], axis=1, inplace=True)
valData.drop(['Koordinater'], axis=1, inplace=True)

# Import LabelEncoder from sklearn for label encoding
from sklearn.preprocessing import LabelEncoder
# Create an instance of LabelEncoder
encoder = LabelEncoder()
# Label encode the 'Mestrostadeparti' column
valData['partiEncoded'] = encoder.fit_transform(valData['Mestrostadeparti'])

# Set target variable as the encoded column
targetVar = valData.partiEncoded
# Drop the 'Mestrostadeparti' column from DataFrame
valData.drop(['Mestrostadeparti'], axis=1, inplace=True)
# Drop the 'partiEncoded' column from DataFrame
valData.drop(['partiEncoded'], axis=1, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(valData, targetVar, test_size=5 / 20)

# Import cross validation scorer
from sklearn.model_selection import cross_val_score

# Import RandomForestClassifier for random forest classification
from sklearn.ensemble import RandomForestClassifier
# Create a RandomForestClassifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model using training data
rf_classifier.fit(X_train, y_train)
# Predict the testing data
predictions = rf_classifier.predict(X_test)

# Compute the cross validation score and print
cv = cross_val_score(rf_classifier, X_train, y_train, cv=3)
print("random forest score:")
print(cv)
print(cv.mean())

# Create a logistic regression model
logreg = LogisticRegression(multi_class='ovr', solver='liblinear')

# Tune Log Reg hyperparameters by using gridsearch
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'penalty': ['l1', 'l2'],
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'fit_intercept': [True, False],
#     'max_iter': [100, 500, 1000],
#     'tol': [1e-4, 1e-3, 1e-2],
#     'class_weight': [None, 'balanced'],
#     'random_state': [None, 42, 2021]
# }
#
# gridsearch = GridSearchCV(logreg, param_grid = param_grid, cv = 3, verbose = False, n_jobs = -1)
# gridsearchedLogReg = gridsearch.fit(X_train, y_train)
#
# cv = cross_val_score(gridsearchedLogReg, X_train, y_train, cv=3)
# print("Grid searched Logistic regression score:")
# print(cv)
# print(cv.mean())

# Fit the logistic regression model with training data
logreg.fit(X_train, y_train)
# Predict the testing data with logistic regression model
logregPredictions = logreg.predict(X_test)
logregPredictions = logregPredictions.T
logregPredictionsDf = pd.DataFrame(logregPredictions, columns=['logregPredictions'])

# Compute the cross validation score for logistic regression model and print
cv = cross_val_score(logreg, X_train, y_train, cv=3)
print("Logistic regression score:")
print(cv)
print(cv.mean())

# Import GradientBoostingClassifier for gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
# Create a gradient boosting classifier
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
# Train the model using training data
gbc.fit(X_train, y_train)

# Compute the cross validation score for gradient boosting classifier and print
cv = cross_val_score(gbc, X_train, y_train, cv=3)
print("Gradient Booster score:")
print(cv)
print(cv.mean())

# Import MultinomialNB for naive bayes classification
from sklearn.naive_bayes import MultinomialNB
# Create a naive bayes classifier
nb_classifier = MultinomialNB()
# Train the model using training data
nb_classifier.fit(X_train, y_train)

# Compute the cross validation score for naive bayes classifier and print
cv = cross_val_score(nb_classifier, X_train, y_train, cv=3)
print("Naive Bayes score:")
print(cv)
print(cv.mean())

from sklearn.ensemble import VotingClassifier

# Create a list of tuples for the classifier. Each tuple contains a string (the name of the model) and an instance of the model.
estimators = [('rf', rf_classifier), ('lr', logreg), ('gb', gbc), ('nb', nb_classifier)]

# Create the ensemble model.
ensemble = VotingClassifier(estimators, voting='hard')

# Fit the ensemble model
ensemble.fit(X_train, y_train)

# Evaluate the ensemble model
cv = cross_val_score(ensemble, X_train, y_train, cv=3)
print("Ensemble score:")
print(cv)
print(cv.mean())

# Merge logistic regression predictions with original DataFrame
X_test.reset_index(inplace=True)
X_test.rename(columns={'index': 'OriginalIndex'}, inplace=True)

logRegPredictionsWithValData = pd.concat([X_test.reset_index(drop=True), logregPredictionsDf.reset_index(drop=True)], axis=1)
valDataMedKommun = valDataMedKommun.reset_index().rename(columns={'index': 'OriginalIndex'})
valDataMedKommun = valDataMedKommun.merge(logRegPredictionsWithValData, how='left')
valDataMedKommun.dropna(inplace=True)

# Decode the 'partiDecoded' column
valDataMedKommun['logregPredictions'] = valDataMedKommun['logregPredictions'].astype(int)
valDataMedKommun['partiDecoded'] = encoder.inverse_transform(valDataMedKommun['logregPredictions'])

# Clean 'Koordinater' column
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° N', '')
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° E', '')
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° S', '')
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° W', '')

# Split the 'Koordinater' column into 'lat' and 'lon'
valDataMedKommun[['lat', 'lon']] = valDataMedKommun['Koordinater'].str.split(',', expand=True)

# Convert 'lat' and 'lon' columns to numeric
valDataMedKommun['lat'] = pd.to_numeric(valDataMedKommun['lat'])
valDataMedKommun['lon'] = pd.to_numeric(valDataMedKommun['lon'])

# Import plotly for creating scatter plot
import plotly.graph_objects as go

fig = go.Figure(data=go.Scattergeo(
    lon = valDataMedKommun['lon'],
    lat = valDataMedKommun['lat'],
    mode = 'markers',
    marker_color = valDataMedKommun['lon'], # change the color attribute to a column in your dataframe
    text = valDataMedKommun[['Kommun', 'Mestrostadeparti', 'partiDecoded']].apply(
        lambda x: f'Kommun: {x[0]}<br> Mest populära parti: {x[1]}<br> Estimat: {x[2]}', axis=1), # Adding hover text
    hoverinfo = 'text' # what information to show when hovering
))

fig.update_geos(
    projection_type="mercator",
    lataxis_range=[55, 70], # the range of latitude
    lonaxis_range=[10, 25]  # the range of longitude
)

fig.update_layout(
    title_text = 'Scatter plot over Sweden',
    geo=dict(
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        countrycolor = "rgb(255, 255, 255)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = True,
        showcountries = True,
        resolution = 50,
    ),
    autosize=True,
)

fig.show()

print("Done")
