import nltk
# nltk.download('all') # one time execution
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd
import re
import string
import contractions
from transformers import BertTokenizer, BertModel

contractions_dict = {"ain't": "are not",
                     "'s":" is",
                     "aren't": "are not",
                     "'fore": "before", 
                     "i've": "i have", 
                     "you've": "you have",
                     "wanna": "want to", 
                     "gotta": "have got to",
                     "gonna": "going to",
                     "ima": "i am going to",
                     "you'll": "you will",
                     "i'll": "i will"}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


class TextPreprocessor:

    def __init__(self, text):    
        self.lyric = text

    # delete punctuations
    def remove_punctuation(self, text):       
        return re.sub(f'[{string.punctuation}]', '', text)

    # lower texts
    def to_lowercase(self, text):       
        return text.lower()
    
     # remove numbers
    def remove_numbers(self, text):     
        return re.sub(r'\d+', '', text)
    
     # remove extra white spaces
    def remove_whitespaces(self, text):      
        return text.strip()
    
     # expand contradiction
    def expand_contraction(self, text, contractions_dict = contractions_dict):     
        expanded_words = []   
        for word in text.split():
            expanded_words.append(contractions.fix(word))  
        expanded_text = ' '.join(expanded_words)

        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, expanded_text)
    
     # delete non-english words
    def del_nonEnglish(self, text):      
        text = re.sub(r'\W+', ' ', text)
        text = text.lower()
        text = text.replace("[^a-zA-Z]", " ")
        word_tokens = word_tokenize(text)
        filtered_word = [w for w in word_tokens if all(ord(c) < 128 for c in w)]
        filtered_word = [w + " " for w in filtered_word]
        return "".join(filtered_word)
    
     # remove stopwords
    def remove_stopwords(self, text):    
        global stop_words
        try:
            word_tokens = word_tokenize(text)
            filtered_word = [w for w in word_tokens if not w in stop_words]
            filtered_word = [w + " " for w in filtered_word]
            return "".join(filtered_word)
        except:
            return np.nan

     #lemmatize
    def normalization(self, text):      
        global lemmatizer
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    
    def preprocess_text(self):
        self.lyric = self.to_lowercase(self.lyric)
        self.lyric = self.expand_contraction(self.lyric)
        self.lyric = self.remove_punctuation(self.lyric)
        self.lyric = self.del_nonEnglish(self.lyric)
        self.lyric = self.remove_stopwords(self.lyric)
        self.lyric = self.normalization(self.lyric)
        self.lyric = self.remove_whitespaces(self.lyric)
        self.lyric = self.remove_numbers(self.lyric)
        
        return self.lyric

