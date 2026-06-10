from sklearn.feature_extraction.text import CountVectorizer

class BoW:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit_transform(self, text_data):
        return self.vectorizer.fit_transform(text_data)

    def transform(self, text_data):
        return self.vectorizer.transform(text_data)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()