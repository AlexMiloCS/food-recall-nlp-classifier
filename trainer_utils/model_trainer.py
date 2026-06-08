from evaluation import SubmissionExporter

class ModelTrainer:
    def __init__(self, extractor, classifier_class):
        self.extractor = extractor
        self.classifier_class = classifier_class

    def run_experiment(self, X, y, encoders, test_indices, export_name="baseline"):
        X_train_vec = self.extractor.fit_transform(X['train'])
        X_val_vec = self.extractor.transform(X['val'])
        X_test_vec = self.extractor.transform(X['test'])

        clf_hazard = self.classifier_class()
        clf_hazard.train(X_train_vec, y['train_hazard'])

        clf_product = self.classifier_class()
        clf_product.train(X_train_vec, y['train_product'])

        f1_hazard = clf_hazard.evaluate(X_val_vec, y['val_hazard'])
        f1_product = clf_product.evaluate(X_val_vec, y['val_product'])
        
        st1_score = (f1_hazard + f1_product) / 2
        print(f"\n ΤΕΛΙΚΟ ST1 SCORE ({export_name}): {st1_score:.4f} <---")

        preds_hazard = clf_hazard.predict(X_test_vec)
        preds_product = clf_product.predict(X_test_vec)

        filepath = f'csv/submission_{export_name}.csv'
                
        exporter = SubmissionExporter()
        
        exporter.export(
            test_indices=test_indices,
            preds_hazard=preds_hazard,
            preds_product=preds_product,
            le_hazard=encoders['hazard'],
            le_product=encoders['product'],
            filepath=filepath
        )
        
        return st1_score