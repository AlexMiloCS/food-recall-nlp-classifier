import os
import numpy as np
from sklearn.metrics import f1_score
from evaluation import SubmissionExporter
from evaluation.create_confusion_matrix import ConfusionMatrixCreator

class ModelPostProcessor:
    def __init__(self, model_prefix, train_df, val_df, test_df, data_dict):
        self.model_prefix = model_prefix
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.data_dict = data_dict
        self.y_true_haz_val = val_df['hazard_label'].values
        self.y_true_prod_val = val_df['product_label'].values

    def save_probabilities(self, haz_val_probs, haz_test_probs, prod_val_probs, prod_test_probs):
        os.makedirs('models/probs', exist_ok=True)        
        np.save(f'models/probs/{self.model_prefix}_haz_val_probs.npy', haz_val_probs)
        np.save(f'models/probs/{self.model_prefix}_haz_test_probs.npy', haz_test_probs)
        np.save(f'models/probs/{self.model_prefix}_prod_val_probs.npy', prod_val_probs)
        np.save(f'models/probs/{self.model_prefix}_prod_test_probs.npy', prod_test_probs)
        

    def evaluate_and_export(self, preds_haz_val, preds_prod_val, preds_haz_test, preds_prod_test):
        f1_haz = f1_score(self.y_true_haz_val, preds_haz_val, average='macro')
        mod_pred_prod = np.where(self.y_true_haz_val == preds_haz_val, preds_prod_val, -1)
        f1_prod = f1_score(self.y_true_prod_val, mod_pred_prod, average='macro')
        final_score = (f1_haz + f1_prod) / 2

        print(f"ST1 SCORE (Validation): {final_score:.4f}")
        print(f"Hazard F1: {f1_haz:.4f} | Product F1: {f1_prod:.4f}")

        exporter = SubmissionExporter()
        filepath = os.path.join('csv', f'{self.model_prefix}_submission.csv')
        exporter.export(
            test_indices=self.test_df.index,
            preds_hazard=preds_haz_test,
            preds_product=preds_prod_test,
            le_hazard=self.data_dict['le_hazard'],
            le_product=self.data_dict['le_product'],
            filepath=filepath
        )

    def plot_hazard_confusion_matrix(self, preds_haz_val):
        cm_creator_haz = ConfusionMatrixCreator(
            train_df=self.train_df,
            val_df=self.val_df,
            target_col='hazard_label',
            class_names=self.data_dict['le_hazard'].classes_
        )
        cm_creator_haz.plot_predictions(
            y_true=self.y_true_haz_val,
            y_pred=preds_haz_val,
            title=f"Confusion Matrix: Standalone {self.model_prefix.upper()} (Hazard)",
            filename=f"cm_{self.model_prefix}_hazard.png"
        )

    def plot_product_confusion_matrix(self, preds_prod_val):
        cm_creator_prod = ConfusionMatrixCreator(
            train_df=self.train_df,
            val_df=self.val_df,
            target_col='product_label',
            class_names=self.data_dict['le_product'].classes_
        )
        cm_creator_prod.plot_predictions(
            y_true=self.y_true_prod_val,
            y_pred=preds_prod_val,
            title=f"Confusion Matrix: Standalone {self.model_prefix.upper()} (Product)",
            filename=f"cm_{self.model_prefix}_product.png",
            figsize=(16, 14),
            annot_size=8
        )