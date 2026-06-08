from sklearn.feature_extraction.text import CountVectorizer

class BoW:
    # Initialize model and parameters
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)

    # Learns vocabulary and transforms
    def fit_transform(self, text_data):
        return self.vectorizer.fit_transform(text_data)

    # Transforms new data only
    def transform(self, text_data):
        return self.vectorizer.transform(text_data)

    # Returns the stored vocabulary
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()