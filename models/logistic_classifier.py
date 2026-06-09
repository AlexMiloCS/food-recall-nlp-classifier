from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

class LogisticClassifier:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        macro_f1 = f1_score(y_test, predictions, average='macro')
        
        print(f" Macro-F1 Score: {macro_f1:.4f}\n")
        print(classification_report(y_test, predictions))
        
        return macro_f1

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)