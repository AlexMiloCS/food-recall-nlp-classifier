import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

class ClassicalTrainer:
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.clf_hazard = None
        self.clf_product = None

    def fit(self, X_train, y_train_hazard, y_train_product):
        hazard_weights = compute_class_weight('balanced', classes=np.unique(y_train_hazard), y=y_train_hazard)
        hazard_weight_dict = dict(enumerate(hazard_weights))

        product_weights = compute_class_weight('balanced', classes=np.unique(y_train_product), y=y_train_product)
        product_weight_dict = dict(enumerate(product_weights))

        self.clf_hazard = LogisticRegression(class_weight=hazard_weight_dict, max_iter=self.max_iter)
        self.clf_product = LogisticRegression(class_weight=product_weight_dict, max_iter=self.max_iter)

        self.clf_hazard.fit(X_train, y_train_hazard)
        self.clf_product.fit(X_train, y_train_product)

        return self

    def evaluate(self, X_val, y_val_hazard, y_val_product):
        preds_hazard = self.clf_hazard.predict(X_val)
        preds_product = self.clf_product.predict(X_val)

        f1_hazard = f1_score(y_val_hazard, preds_hazard, average='macro')

        correct_mask = (preds_hazard == y_val_hazard)
        if correct_mask.sum() == 0:
            f1_product_cond = 0.0
        else:
            f1_product_cond = f1_score(y_val_product[correct_mask], preds_product[correct_mask], average='macro')

        st1_score = (f1_hazard + f1_product_cond) / 2

        print(f"Validation -> Hazard F1: {f1_hazard:.4f} | Product F1: {f1_product_cond:.4f} | ST1 Score: {st1_score:.4f}")
        
        return st1_score

    def predict(self, X_test):
        preds_hazard = self.clf_hazard.predict(X_test)
        preds_product = self.clf_product.predict(X_test)
        return preds_hazard, preds_product