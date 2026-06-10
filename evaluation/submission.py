import pandas as pd

class SubmissionExporter:
    def __init__(self):
        pass

    def export(self, test_indices, preds_hazard, preds_product, le_hazard, le_product, filepath):
        predicted_hazards = le_hazard.inverse_transform(preds_hazard)
        predicted_products = le_product.inverse_transform(preds_product)

        submission = pd.DataFrame({
            'hazard-category': predicted_hazards,
            'product-category': predicted_products
        }, index=test_indices)

        submission.to_csv(filepath)
        print(f"Το αρχείο '{filepath}' δημιουργήθηκε επιτυχώς ({len(submission)} γραμμές).")