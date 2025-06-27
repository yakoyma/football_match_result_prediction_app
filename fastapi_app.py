"""
===============================================================================
Multiclass Classification Project: Prediction of the football match result
with a Machine Learning (ML) model and FastAPI
===============================================================================
"""
# Standard library
import platform

# Other libraries
import pandas as pd
import numpy as np
import spacy
import pickle
import fastapi


from fastapi import FastAPI


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('SpaCy: {}'.format(spacy.__version__))
print('FastAPI: {}'.format(fastapi.__version__))



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

    # Instantiate the NLP model
    nlp = spacy.load(name='xx_ent_wiki_sm')
    nlp.add_pipe(factory_name='sentencizer')

    # Cleanse texts
    HomeTeam = HomeTeam.strip()
    AwayTeam = AwayTeam.strip()

    # Check wether inputs are valid
    HomeTeam_tokens_length = len(nlp(HomeTeam))
    AwayTeam_tokens_length = len(nlp(AwayTeam))
    if (HomeTeam_tokens_length > 0 and AwayTeam_tokens_length > 0 and
        HomeTeam != AwayTeam and HomeTeamScore >= 0 and AwayTeamScore >= 0):

        # Create dataset with input data
        X = pd.DataFrame(
            data={HomeTeam: [HomeTeamScore], AwayTeam: [AwayTeamScore]})

        # Load the trained ML model
        model_path = open('models/model.pkl', 'rb')
        model_pickle = pickle.load(model_path)
        model = model_pickle['model']

        # Make prediction
        prediction = model.predict(np.array(X))

        # Load encoder
        encoder_path = open('models/encoder.pkl', 'rb')
        encoder_pickle = pickle.load(encoder_path)
        encoder = encoder_pickle['encoder']
        label = encoder.inverse_transform(prediction)[0]

        # Display the result
        results = {
            f'{HomeTeam} wins': 'Home team wins',
            f'{AwayTeam} wins': 'Away team wins',
            'The match ends in a draw': 'Draw match'
        }
        result = next(key for key, value in results.items() if label == value)
        answer = (f'The result of the football match between {HomeTeam} and '
                  f'{AwayTeam} is: {result}.')
        response = {'Parameters': params, 'Response': answer}
    else:
        response = {
            'Parameters': params,
            'Error message': 'Invalid input data. Please complete all fields '
                             'correctly.'
        }
    return response
