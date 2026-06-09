from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha, fit_prior=False)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        macro_f1 = f1_score(y_test, predictions, average='macro')
        print(f"\n--- Naive Bayes Evaluation ---")
        print(f"Macro-F1 Score: {macro_f1:.4f}\n")
        return macro_f1