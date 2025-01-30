"""
===============================================================================
Multiclass Classification Project: Prediction of the football match result
using Gradio
===============================================================================

This file is organised as follows:
Prediction using Gradio application
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
Prediction using Gradio application
===============================================================================
"""
def get_prediction(HomeTeam: str, AwayTeam: str, HomeTeamScore: int,
                   AwayTeamScore: int):
    """This function predicts the football match result using Machine
    Learning model.

    Args:
        HomeTeam (str): the name of home team
        AwayTeam (str): the name of away team
        HomeTeamScore (int): the score of home team
        AwayTeamScore (int): the score of away team

    Returns:
        response (str): the predicted result of the match
    """

    # Create dataset with inputs data
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

    result = (f'The result of the football match between {HomeTeam} and '
              f'{AwayTeam} is: {label}.')
    return result


# Instantiate the app
app = gr.Interface(
    fn=get_prediction,
    inputs=['text', 'text', 'number', 'number'],
    outputs='text',
    title='Prediction of Football Match Result'
)



if __name__ == '__main__':
    app.launch()
