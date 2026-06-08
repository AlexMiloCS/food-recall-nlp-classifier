import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self):
        # Ρυθμίσεις εμφάνισης για επαγγελματικά γραφήματα
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [12, 6]

    def plot_hazard_distribution(self, df):
        plt.figure(figsize=(12, 6))
        counts = df['hazard-category'].value_counts()
        
        sns.barplot(x=counts.values, y=counts.index, palette="viridis")
        plt.title("Κατανομή των Hazard Categories (Train Set)", fontsize=16, pad=15)
        plt.xlabel("Αριθμός Δειγμάτων", fontsize=12)
        plt.ylabel("Κατηγορία Κινδύνου", fontsize=12)
        
        plt.tight_layout()
        plt.show()

    def plot_product_distribution(self, df):
        plt.figure(figsize=(12, 8)) 
        counts = df['product-category'].value_counts()
        
        sns.barplot(x=counts.values, y=counts.index, palette="magma")
        plt.title("Κατανομή των Product Categories (Long Tail)", fontsize=16, pad=15)
        plt.xlabel("Αριθμός Δειγμάτων", fontsize=12)
        plt.ylabel("Κατηγορία Προϊόντος", fontsize=12)
        
        plt.tight_layout()
        plt.show()