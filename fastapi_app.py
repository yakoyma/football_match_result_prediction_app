"""
===============================================================================
Multiclass Classification Project: Prediction of the football match result
using FastAPI
===============================================================================

This file is organised as follows:
Prediction using FastAPI application
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
Prediction using FastAPI application
===============================================================================
"""
# Instantiate the app
app = FastAPI()


@app.get('/predict/')
async def get_prediction(HomeTeam: str, AwayTeam: str, HomeTeamScore: int,
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
        # Create dataset with inputs data
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

        result = (f'The result of the football match between {HomeTeam} and '
                  f'{AwayTeam} is: {label}.')
        return {'Parameters': params, 'Result': result}
