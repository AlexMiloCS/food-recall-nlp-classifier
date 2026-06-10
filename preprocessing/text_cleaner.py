import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextCleaner:
    
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        
        words = word_tokenize(text)
        
        cleaned_words = []
        for word in words:
            if word not in self.stop_words:
                lemma_word = self.lemmatizer.lemmatize(word)
                cleaned_words.append(lemma_word)
                
        final_text = ' '.join(cleaned_words)
        final_text = re.sub(r'\s+', ' ', final_text).strip()
        
        return final_text