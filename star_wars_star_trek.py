#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict(message):
    
    model=load_model('news_in.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        
        tokenizer2 = pickle.load(handle)
        x_1 = tokenizer2.texts_to_sequences([message])
        x_1 = pad_sequences(x_1, maxlen=500)
        predictions = model.predict(x_1)[0][0]
    return predictions


title_ = '''
<style>
h2 {
    color: white;
    font-size: 50px; 
    font-family: 'News Gothic';   
}
</style>


  <h2>Testing</h2>

'''
st.markdown(title_, unsafe_allow_html=True)
st.title('News/Not News Sentiment Analyzer')
message = st.text_area('Enter sentence','Type Here ..')

if st.button('Analyze'):
    with st.spinner('Analyzing the text …'):
        prediction = predict(message)
    
    if prediction > 0.6:
        st.success('Tagged as News with {:.2f}% confidence'.format(prediction*100))
        st.balloons()
    elif prediction < 0.4:
        st.error('Tagged as Not News with {:.2f}% confidence'.format((1-prediction)*100))
    else:
        st.warning('Not sure! Try to add some more words')
#https://www.intellectualtakeout.org/assets/3/28/star_wars_vs_star_trek_by_hapo57.jpg
page_bg_img = '''
<style>
body {
background-image: url("https://wallpapercave.com/wp/wp2610854.png");
background-size: cover;
}
</style>
'''



st.markdown(page_bg_img, unsafe_allow_html=True)