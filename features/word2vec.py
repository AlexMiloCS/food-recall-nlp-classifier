import numpy as np
from gensim.models import Word2Vec
import gensim.downloader as api

class Word2VecExtractor:
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X):
        sentences = [text.split() for text in X]
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        return self

    def document_vector(self, tokens):
        if self.model is None:
            raise ValueError("Το μοντέλο Word2Vec δεν έχει εκπαιδευτεί. Κάλεσε την fit() πρώτα.")
            
        valid_words = [word for word in tokens if word in self.model.wv.key_to_index]
        if not valid_words:
            return np.zeros(self.vector_size)
            
        return np.mean(self.model.wv[valid_words], axis=0)

    def transform(self, X):
        sentences = [text.split() for text in X]
        return np.array([self.document_vector(tokens) for tokens in sentences])
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class PreTrainedWord2Vec:
    def __init__(self, model_name='word2vec-google-news-300'):
        self.wv = api.load(model_name)
        self.vector_size = self.wv.vector_size

    def _get_document_vector(self, text):
        words = str(text).split() 
        valid_words = [word for word in words if word in self.wv]

        if valid_words:
            return np.mean(self.wv[valid_words], axis=0)
        else:
            return np.zeros(self.vector_size)

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        vectors = [self._get_document_vector(text) for text in X]
        return np.array(vectors)