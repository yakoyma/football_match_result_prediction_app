"""
===============================================================================
Multiclass Classification Project: Prediction of the football match result
using an API application
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
4. Prediction using the FastAPI application
"""
# Standard libraries
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import pandas as pd
import sklearn
import fastapi


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             multilabel_confusion_matrix,
                             classification_report)
from fastapi import FastAPI


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('FastAPI: {}'.format(fastapi.__version__))



# Constant
SEED = 0

# Set the maximum number of rows to display by Pandas
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Data Analysis
===============================================================================
"""
# Load the dataset
print('\n\n\nLoad the dataset: ')
INPUT_CSV_1 = 'https://www.football-data.co.uk/mmz4281/2223/E0.csv'
INPUT_CSV_2 = 'https://www.football-data.co.uk/mmz4281/2223/E1.csv'
INPUT_CSV_3 = 'https://www.football-data.co.uk/mmz4281/2223/E2.csv'
raw_dataset_e0 = pd.read_csv(INPUT_CSV_1)
raw_dataset_e1 = pd.read_csv(INPUT_CSV_2)
raw_dataset_e2 = pd.read_csv(INPUT_CSV_3)
raw_dataset = pd.concat([raw_dataset_e0, raw_dataset_e1, raw_dataset_e2])
raw_dataset = raw_dataset.reset_index(inplace=False, drop=True)

# Display the raw dataset's dimensions
print('\nDimensions of the raw dataset: {}'.format(raw_dataset.shape))

# Display the raw dataset's information
print('\nInformation about the raw dataset:')
print(raw_dataset.info())

# Description of the raw dataset
print('\nDescription of the raw dataset:')
print(raw_dataset.describe(include='all'))

# Display the head and the tail of the raw dataset
print(f'\nRaw dataset shape: {raw_dataset.shape}')
print(raw_dataset.info())
print(pd.concat([raw_dataset.head(150), raw_dataset.tail(150)]))


# Cleanse the dataset
subset  = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG']
dataset = raw_dataset[subset]

# Management of duplicates
print('\n\nManagement of duplicates:')
duplicate = dataset[dataset.duplicated(subset=subset, keep='first')]
print('Dimensions of the duplicates dataset: {}'.format(duplicate.shape))
print(pd.concat([duplicate.head(), duplicate.tail()]))
if duplicate.shape[0] > 0:
    dataset = dataset.drop_duplicates(keep='first')
    dataset.reset_index(inplace=True, drop=True)

dataset['FTR'] = dataset['FTR'].map(
    {'H': 'Home team wins', 'A': 'Away team wins', 'D': 'Draw match'})
dataset = dataset.rename(columns={
    'FTR': 'Result', 'FTHG': 'HomeTeamScore', 'FTAG': 'AwayTeamScore'})

# Display the dataset's dimensions
print('\nDimensions of the dataset: {}'.format(dataset.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(dataset.info())

# Description of the dataset
print('\nDescription of the dataset:')
print(dataset.describe(include='all'))

# Display the head and the tail of the dataset
print(f'\nDataset shape: {dataset.shape}')
print(dataset.info())
print(pd.concat([dataset.head(150), dataset.tail(150)]))



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
# Feature selection
features = ['HomeTeamScore', 'AwayTeamScore']
X = dataset[features]
y = dataset['Result']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Display the head and the tail of the train dataset
print(f'\n\n\nTrain dataset shape: {X_train.shape}')
print(X_train.info())
print(pd.concat([X_train.head(150), X_train.tail(150)]))

# Display the head and the tail of the test dataset
print(f'\nTest dataset shape: {X_test.shape}')
print(X_test.info())
print(pd.concat([X_test.head(150), X_test.tail(150)]))


# Encode the label
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
print(f'\nTrain label shape: {y_train.shape}')
print(f'Test label shape: {y_test.shape}')



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
# Instantiate the model
model = LogisticRegression(class_weight='balanced', random_state=SEED)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print('\n\n\nAccurcay: {:.3f}'.format(
    accuracy_score(y_test, y_pred)))
print('Balanced Accurcay: {:.3f}'.format(
    balanced_accuracy_score(y_test, y_pred)))
print('Multilabel Confusion Matrix:\n{}'.format(
    multilabel_confusion_matrix(y_test, y_pred)))
print('Classification Report:\n{}'.format(
    classification_report(y_test, y_pred)))



"""
===============================================================================
4. Prediction using the FastAPI application
===============================================================================
"""
# Instantiate the app
app = FastAPI()


@app.get('/predict/')
async def predict(HomeTeam: str, AwayTeam: str, HomeTeamScore: int,
                  AwayTeamScore: int):
    params = {
        'Home team': HomeTeam,
        'Away team': AwayTeam,
        'Home team score': HomeTeamScore,
        'Away team score': AwayTeamScore
    }
    if HomeTeam == AwayTeam or HomeTeamScore < 0 or AwayTeamScore < 0:
        return {
            'Parameters': params,
            'Error message': 'Invalid input data'
        }
    else:
        data = [HomeTeamScore, AwayTeamScore]
        prediction = model.predict([data])
        return {
            'Parameters': params,
            'Result': encoder.inverse_transform(prediction)[0],
        }
