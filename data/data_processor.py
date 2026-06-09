import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from preprocessing.text_cleaner import TextCleaner

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DataProcessor:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.le_hazard = LabelEncoder()
        self.le_product = LabelEncoder()

    def _prepare_dataset(self, df, tokenize=False):
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        df['full_text'] = df['title'] + " " + df['text']
        df['clean_text'] = df['full_text'].apply(self.cleaner.clean)
        
        if tokenize:
            df['tokens'] = df['clean_text'].apply(word_tokenize)
            
        return df

    def process_data(self, train_path, val_path, test_path=None, tokenize=False):

        train_df = pd.read_csv(train_path, index_col=0)
        val_df = pd.read_csv(val_path, index_col=0)

        train_df = self._prepare_dataset(train_df, tokenize)
        val_df = self._prepare_dataset(val_df, tokenize)

        train_df['hazard_label'] = self.le_hazard.fit_transform(train_df['hazard-category'])
        train_df['product_label'] = self.le_product.fit_transform(train_df['product-category'])

        val_df['hazard_label'] = self.le_hazard.transform(val_df['hazard-category'])
        val_df['product_label'] = self.le_product.transform(val_df['product-category'])

        result = {
            'train_df': train_df,
            'val_df': val_df,
            'le_hazard': self.le_hazard,
            'le_product': self.le_product
        }

        if test_path:
            test_df = pd.read_csv(test_path, index_col=0)
            test_df = self._prepare_dataset(test_df, tokenize)
            result['test_df'] = test_df

        return result
    
    def prepare_baseline_data(self, train_path='csv/train.csv', val_path='csv/valid.csv', test_path='csv/test.csv'):
        data = self.process_data(train_path, val_path, test_path, tokenize=False)

        X = {
            'train': data['train_df']['clean_text'],
            'val': data['val_df']['clean_text'],
            'test': data['test_df']['clean_text']
        }
        
        y = {
            'train_hazard': data['train_df']['hazard_label'].values,
            'train_product': data['train_df']['product_label'].values,
            'val_hazard': data['val_df']['hazard_label'].values,
            'val_product': data['val_df']['product_label'].values
        }
        
        encoders = {
            'hazard': data['le_hazard'],
            'product': data['le_product']
        }
        
        test_indices = data['test_df'].index
        
        return X, y, encoders, test_indices