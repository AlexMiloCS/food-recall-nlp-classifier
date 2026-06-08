import numpy as np
from sklearn.metrics import f1_score

class KaggleScorer:
    def __init__(self):
        pass

    # Calculates the official ST1 metric
    def evaluate(self, y_true_hazard, y_pred_hazard, y_true_product, y_pred_product):
        # Calculate F1 for hazard
        f1_hazard = f1_score(y_true_hazard, y_pred_hazard, average='macro')

        # Find correct hazard predictions
        correct_mask = (y_pred_hazard == y_true_hazard)

        # Calculate conditional F1 for product
        if correct_mask.sum() == 0:
            f1_product_cond = 0.0
        else:
            f1_product_cond = f1_score(
                y_true_product[correct_mask], 
                y_pred_product[correct_mask], 
                average='macro'
            )

        # Final ST1 Score
        st1_score = (f1_hazard + f1_product_cond) / 2
        
        print("\n--- Kaggle Evaluation ---")
        print(f"F1 Hazard: {f1_hazard:.4f}")
        print(f"F1 Product (Conditional): {f1_product_cond:.4f}")
        print(f" Official ST1 Score: {st1_score:.4f}")
        print("----------------------------\n")
        
        return st1_score