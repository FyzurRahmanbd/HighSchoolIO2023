import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset and preprocess 'Label'
df = pd.read_csv('data.csv')
df['Label'] = df['Label'].map({'FAKE': 0, 'REAL': 1})
df = df.drop('ID', axis=1)

# Separate text and labels for training
X = df['Text'].values
y = df['Label'].values

# Vectorize the text data using TF-IDF from scikit-learn
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Create and train the Logistic Regression model
logreg_classifier = LogisticRegression(random_state=17)
logreg_classifier.fit(X_tfidf, y)

# Function to predict if a text is real or fake
def predict_text(text):
    # Vectorize the input text using the same vectorizer
    text_tfidf = vectorizer.transform([text])

    # Make a prediction
    prediction = logreg_classifier.predict(text_tfidf)

    # Return the predicted label
    return prediction[0]


# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1('Fake News Detection App'),

    html.Div([
        html.Label('Enter a text:'),
        dcc.Textarea(id='input-text', value='This is a sample text.'),
        html.Button('Predict', id='predict-button'),
    ]),

    html.Div(id='prediction-result', style={'marginTop': 20, 'fontWeight': 'bold'})
])

# Define the callback to update the prediction result
@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-text', 'value')]
)
def update_prediction(n_clicks, input_text):
    if n_clicks is None:
        return ''
    else:
        prediction = predict_text(input_text)
        result = 'REAL' if prediction == 1 else 'FAKE'
        return f'The model predicts: {result}'

if __name__ == '__main__':
    app.run_server(debug=True)
