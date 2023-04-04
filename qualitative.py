import pandas as pd
import numpy as np
import re
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import string

from fastapi import FastAPI


app = FastAPI()
#templates = Jinja2Templates(directory="templates")
tokenizer = Tokenizer(num_words=10000)

STOPWORDS = stopwords.words('english')
PUNCTUATION = string.punctuation

def remove_punctuation(series):
    no_punctuation = "".join([word for word in series if word not in PUNCTUATION])
    return no_punctuation

def remove_extra_white_spaces(series):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string = series)
    return without_sc

def remove_stopwords(text):
    no_stopwords = []    
    tokens = word_tokenize(text)
    
    for i in range(len(tokens)):
        if tokens[i] not in STOPWORDS:
            no_stopwords.append(tokens[i])
            
    return " ".join(no_stopwords)

# Create a route to handle incoming post requests
@app.post("/predict")
async def predict(text: str):
    # Clean the text
    df = pd.DataFrame([text], columns = ['text'])

    df['text'] = df['text'].apply(remove_punctuation)

    df['text'] = df['text'].apply(remove_extra_white_spaces)

    df['text'] = df['text'].apply(remove_stopwords)

    text = df.text
    
    # Load the trained model
    model = tensorflow.keras.models.load_model('SGD_model.h5')
    
    # Tokenize the text
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=100)

    labels = ['Quantitative','Qualitative']

    # Make predictions
    predictions = model.predict(padded_sequences)
    predictions = np.argmax(predictions,axis=1)
    
    # Return the predictions
    return {'predictions': predictions.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

