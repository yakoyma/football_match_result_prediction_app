"""
===============================================================================
Multiclass Classification Project: Prediction of the football match result
with a Machine Learning (ML) model and FastAPI
===============================================================================

This file is organised as follows:
Prediction application with FastAPI
"""
# Standard libraries
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import pandas as pd
import numpy as np
import pickle
import fastapi


from fastapi import FastAPI


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('FastAPI: {}'.format(fastapi.__version__))



"""
===============================================================================
Prediction application with FastAPI
===============================================================================
"""
# Instantiate the app
app = FastAPI()


@app.get('/predict/')
async def get_prediction(HomeTeam: str, AwayTeam: str, HomeTeamScore: int,
                         AwayTeamScore: int):

    # Set up parameters
    params = {
        'Home team': HomeTeam,
        'Away team': AwayTeam,
        'Home team score': HomeTeamScore,
        'Away team score': AwayTeamScore
    }


    # Check wether input data are valid
    if (HomeTeam and AwayTeam and HomeTeam != AwayTeam and HomeTeamScore >= 0
        and AwayTeamScore >= 0):

        # Create dataset with input data
        X = pd.DataFrame(
            data={HomeTeam: [HomeTeamScore], AwayTeam: [AwayTeamScore]})


        # Load the model
        path = open('models/model.pkl', 'rb')
        model_pickle = pickle.load(path)
        model = model_pickle['model']

        # Make prediction
        prediction = model.predict(np.array(X))

        # Load encoder
        path = open('models/encoder.pkl', 'rb')
        encoder_pickle = pickle.load(path)
        encoder = encoder_pickle['encoder']
        label = encoder.inverse_transform(prediction)[0]


        # Display the result
        results = {
            f'{HomeTeam} wins': 'Home team wins',
            f'{AwayTeam} wins': 'Away team wins',
            'The match ends in a draw': 'Draw match'
        }
        result = next(key for key, value in results.items() if label == value)
        response = (f'The result of the football match between {HomeTeam} and '
                    f'{AwayTeam} is: {result}.')
        responses = {'Parameters': params, 'Response': response}

    else:

        responses = {
            'Parameters': params,
            'Error message': 'Invalid input data. Please complete all fields '
                             'correctly.'
        }
    return responses
