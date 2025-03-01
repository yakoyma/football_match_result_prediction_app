"""
===============================================================================
Multiclass Classification Project: Prediction of the football match result
with a Machine Learning (ML) model and Gradio
===============================================================================

This file is organised as follows:
Prediction application with Gradio
"""
# Standard libraries
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import numpy as np
import pandas as pd
import pickle
import gradio as gr


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Gradio: {}'.format(gr.__version__))



"""
===============================================================================
Prediction application with Gradio
===============================================================================
"""
def get_prediction(HomeTeam: str, AwayTeam: str, HomeTeamScore: int,
                   AwayTeamScore: int):
    """This function predicts the football match result with a ML model.

    Args:
        HomeTeam (str): the name of home team
        AwayTeam (str): the name of away team
        HomeTeamScore (int): the score of home team
        AwayTeamScore (int): the score of away team

    Returns:
        response (str): the predicted result of the match
    """

    # Check wether input data are valid
    if (HomeTeam and AwayTeam and HomeTeam != AwayTeam and HomeTeamScore >= 0
        and AwayTeamScore >= 0):

        # Create dataset with input data
        X = pd.DataFrame(data={
            HomeTeam: [HomeTeamScore], AwayTeam: [AwayTeamScore]})


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

    else:

        response = 'Invalid input data. Please complete all fields correctly.'
    return response



# Instantiate the app
app = gr.Interface(
    fn=get_prediction,
    inputs=['text', 'text', 'number', 'number'],
    outputs='text',
    title='Application for Predicting the Result of a Football Match'
)



if __name__ == '__main__':
    app.launch()
