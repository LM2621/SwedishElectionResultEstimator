from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Preprocessing of data:
# All data is collected from SCB (Statistiska central byrån)
# Some feature engineering in the proprocessing ( for example calculating percentage from the total,
# calculating most voted party per municipality, etc)
# Normalization of columns, like average salary, has been tried but did not improve score
# Plots the results as an interactive map, showing the municipality, truth and estimate
# Currently best performance of this model is 0.75

valData = pd.read_csv("/home/max/Documents/Data science projects/ValEstimator/MasterSheetWithMunicipalityCoordinates.csv")
valDataMedKommun = valData.copy()
valData.drop(['Kommun'], axis=1, inplace=True)
valData.drop(['Koordinater'], axis=1, inplace=True)
# Create an instance of LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# Label encode the 'category' column
valData['partiEncoded'] = encoder.fit_transform(valData['Mestrostadeparti'])
targetVar = valData.partiEncoded
valData.drop(['Mestrostadeparti'], axis=1, inplace=True)

valData.drop(['partiEncoded'], axis=1, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(valData,
                                                    targetVar,
                                                    test_size=5 / 20)
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

# Fit the model to the training data
logreg.fit(X_train, y_train)
logregPredictions = logreg.predict(X_test)
logregPredictions = logregPredictions.T
logregPredictionsDf = pd.DataFrame(logregPredictions, columns=['logregPredictions'])

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

# Merge the logistic regressino predictions with the original dataframe containing municipality
X_test.reset_index(inplace=True)
X_test.rename(columns={'index': 'OriginalIndex'}, inplace=True)

logRegPredictionsWithValData = pd.concat([X_test.reset_index(drop=True), logregPredictionsDf.reset_index(drop=True)], axis=1)
valDataMedKommun = valDataMedKommun.reset_index().rename(columns={'index': 'OriginalIndex'})

valDataMedKommun = valDataMedKommun.merge(logRegPredictionsWithValData, how='left')
valDataMedKommun.dropna(inplace=True)

#Decode the political party column
valDataMedKommun['logregPredictions'] = valDataMedKommun['logregPredictions'].astype(int)
valDataMedKommun['partiDecoded'] = encoder.inverse_transform(valDataMedKommun['logregPredictions'])

# Assuming df is your DataFrame and 'coordinates' is the column with the combined lat/lon data
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° N', '')
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° E', '')
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° S', '')
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° W', '')

# Split the 'coordinates' column into two new columns 'latitude' and 'longitude'
valDataMedKommun[['lat', 'lon']] = valDataMedKommun['Koordinater'].str.split(',', expand=True)

# Convert the new columns to numeric type
valDataMedKommun['lat'] = pd.to_numeric(valDataMedKommun['lat'])
valDataMedKommun['lon'] = pd.to_numeric(valDataMedKommun['lon'])

import plotly.graph_objects as go

fig = go.Figure(data=go.Scattergeo(
    lon = valDataMedKommun['lon'],
    lat = valDataMedKommun['lat'],
    mode = 'markers',
    marker_color = valDataMedKommun['lon'], # change the color attribute to a column in your dataframe
    text = valDataMedKommun[['Kommun', 'Mestrostadeparti', 'partiDecoded']].apply(
        lambda x: f'Kommun: {x[0]}<br> Mest populära parti: {x[1]}<br> Estimat: {x[2]}', axis=1), # Adding hover text
    # text2 = valDataMedKommun[['partiDecoded']].apply(lambda x: f'{x[0]}<br>', axis=1), # Adding hover text
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
