from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score

class SVMClassifier:
    def __init__(self, C=1.0):
        self.model = LinearSVC(C=C, class_weight='balanced', max_iter=2000, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        macro_f1 = f1_score(y_test, predictions, average='macro')
        
        print(f"Macro-F1 Score: {macro_f1:.4f}\n")
        print(classification_report(y_test, predictions, zero_division=0))
        
        return macro_f1


class FinalSVMClassifier:
    def __init__(self, C=5.0):
        base_svm = LinearSVC(C=C, class_weight='balanced', max_iter=2000, random_state=42)
        self.model = CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)