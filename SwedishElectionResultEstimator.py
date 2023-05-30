from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the preprocessed data from the csv file.
valData = pd.read_csv("/home/max/Documents/Data science projects/ValEstimator/MasterSheetWithMunicipalityCoordinates.csv")

# Copy the dataframe and drop the 'Kommun' and 'Koordinater' columns from the original.
valDataMedKommun = valData.copy()
valData.drop(['Kommun'], axis=1, inplace=True)
valData.drop(['Koordinater'], axis=1, inplace=True)

# Create a LabelEncoder object.
encoder = LabelEncoder()

# Fit the encoder to the 'Mestrostadeparti' column, transforming the string categories to integers.
valData['partiEncoded'] = encoder.fit_transform(valData['Mestrostadeparti'])
targetVar = valData.partiEncoded
valData.drop(['Mestrostadeparti'], axis=1, inplace=True)

# Remove the encoded column from the dataframe.
valData.drop(['partiEncoded'], axis=1, inplace=True)

# Split the data into a training set and a testing set.
# The test set will be 25% of the total data.
X_train, X_test, y_train, y_test = train_test_split(valData,
                                                    targetVar,
                                                    test_size=5 / 20)

# Import the random forest classifier from scikit-learn, set the number of estimators (trees) to 100, and fit the model.
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Import cross validation score, perform cross validation on the model, and print the results.
from sklearn.model_selection import cross_val_score
cv = cross_val_score(rf_classifier, X_train, y_train, cv=3)
print("random forest score:")
print(cv)
print(cv.mean())

# Fit a logistic regression model on the training data. Perform cross validation, and print the results.
logreg = LogisticRegression(multi_class='ovr', solver='liblinear')
logreg.fit(X_train, y_train)
cv = cross_val_score(logreg, X_train, y_train, cv=3)
print("Logistic regression score:")
print(cv)
print(cv.mean())

# Fit a gradient boosting classifier on the training data, perform cross validation, and print the results.
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc.fit(X_train, y_train)
cv = cross_val_score(gbc, X_train, y_train, cv=3)
print("Gradient Booster score:")
print(cv)
print(cv.mean())

# Fit a naive bayes classifier on the training data, perform cross validation, and print the results.
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
cv = cross_val_score(nb_classifier, X_train, y_train, cv=3)
print("Naive Bayes score:")
print(cv)
print(cv.mean())

from sklearn.ensemble import VotingClassifier

# Create a list of the classifiers, containing a string (the name of the model) and the instance of the model.
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


# Predict the classes on the test set.
logregPredictions = logreg.predict(X_test)
logregPredictionsDf = pd.DataFrame(logregPredictions, columns=["logregPredictions"])

# Reset the index in the test data and rename the old index.
X_test.reset_index(inplace=True)
X_test.rename(columns={'index': 'OriginalIndex'}, inplace=True)

# Concatenate the predicted labels to the test data dataframe.
X_test = pd.concat([X_test, logregPredictionsDf], axis=1)

# Concatenate the test data dataframe to the original data dataframe that includes 'Kommun' and 'Koordinater'.
valDataMedKommun = pd.concat([valDataMedKommun, X_test], axis=1)

# Drop non-predicted rows.
valDataMedKommun.dropna(inplace=True)

# Decode the political party column.
valDataMedKommun['logregPredictions'] = valDataMedKommun['logregPredictions'].astype(int)
valDataMedKommun['partiDecoded'] = encoder.inverse_transform(valDataMedKommun['logregPredictions'])

# Clean up 'Koordinater' column and split it into 'lat' and 'lon' columns.
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° N', '')
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° E', '')
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° S', '')
valDataMedKommun['Koordinater'] = valDataMedKommun['Koordinater'].str.replace('° W', '')
valDataMedKommun[['lat', 'lon']] = valDataMedKommun['Koordinater'].str.split(',', expand=True)
valDataMedKommun['lat'] = pd.to_numeric(valDataMedKommun['lat'])
valDataMedKommun['lon'] = pd.to_numeric(valDataMedKommun['lon'])

# Create a scatter plot of the municipalities of Sweden with the model's predictions and actual values.
import plotly.graph_objects as go
fig = go.Figure(data=go.Scattergeo(
    lon = valDataMedKommun['lon'],
    lat = valDataMedKommun['lat'],
    mode = 'markers',
    marker_color = valDataMedKommun['lon'],
    text = valDataMedKommun[['Kommun', 'Mestrostadeparti', 'partiDecoded']].apply(
        lambda x: f'Kommun: {x[0]}<br> Mest populära parti: {x[1]}<br> Estimat: {x[2]}', axis=1),
    hoverinfo = 'text'
))
fig.update_geos(
    projection_type="mercator",
    lataxis_range=[55, 70],
    lonaxis_range=[10, 25]
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
