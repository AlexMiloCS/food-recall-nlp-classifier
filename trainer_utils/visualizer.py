import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class ModelVisualizer:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, title, cmap='Blues'):
        """
        Υπολογιζει και ζωγραφιζει το Confusion Matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=classes, yticklabels=classes)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_classification_report(y_true, y_pred, classes, title):
        """
        Τυπωνει το αναλυτικο Classification Report (Precision, Recall, F1-Score).
        """
        print(f"\n{'='*60}")
        print(f"{title.center(60)}")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))