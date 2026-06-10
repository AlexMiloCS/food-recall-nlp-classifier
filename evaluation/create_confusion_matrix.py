import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ConfusionMatrixCreator:
    def __init__(self, train_df, val_df, target_col, class_names, img_dir="images"):
        self.train_df = train_df
        self.val_df = val_df
        self.target_col = target_col
        self.class_names = class_names
        self.img_dir = img_dir
        os.makedirs(self.img_dir, exist_ok=True)

    def train_and_plot(self, classifier, extractor, title, filename, figsize=(12, 10), annot_size=12):
        X_train = extractor.fit_transform(self.train_df['clean_text'])
        X_val = extractor.transform(self.val_df['clean_text'])
        
        y_train = self.train_df[self.target_col].values
        y_val = self.val_df[self.target_col].values
        
        classifier.train(X_train, y_train)
        preds = classifier.predict(X_val)
        
        self.plot_predictions(y_val, preds, title, filename, figsize, annot_size)

    def plot_predictions(self, y_true, y_pred, title, filename, figsize=(12, 10), annot_size=12):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    annot_kws={"size": annot_size})
        
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        
        filepath = os.path.join(self.img_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.show()