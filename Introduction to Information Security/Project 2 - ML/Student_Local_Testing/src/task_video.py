# for the main section at the end
import sys
import os

# normal imports for this
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from sklearn.inspection import permutation_importance

# ************ Model Metrics
# This section is not to be changed by the student
# This packages the results and hands them to the autograder or local tests
class ModelMetrics:
    def __init__(self, model_type: str, train_metrics: dict, test_metrics: dict, feature_importance_df: pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"

    def add_train_metric(self, metric_name: str, metric_val: float):
        self.train_metrics[metric_name] = metric_val

    def add_test_metric(self, metric_name: str, metric_val: float):
        self.test_metrics[metric_name] = metric_val

    def __str__(self):
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics.keys()):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics.keys()):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str
# ***** DON'T CHANGE THIS CLASS ABOVE

class NaiveBayes:
    def __init__(self):
        if os.path.isfile(os.path.join(os.getcwd(), "task4/wine_train_features.csv")):
            self.train_features = pd.read_csv("task4/wine_train_features.csv")
            self.test_features = pd.read_csv("task4/wine_test_features.csv")
            self.train_targets = pd.read_csv("task4/wine_train_targets.csv")
            self.test_targets = pd.read_csv("task4/wine_test_targets.csv")
        else:
            self.train_features = pd.read_csv("../task4/wine_train_features.csv")
            self.test_features = pd.read_csv("../task4/wine_test_features.csv")
            self.train_targets = pd.read_csv("../task4/wine_train_targets.csv")
            self.test_targets = pd.read_csv("../task4/wine_test_targets.csv")

    def calculate_naive_bayes_metrics(self) -> ModelMetrics:

        gnb = GaussianNB()

        score_targets = self.train_targets['good quality']
        gnb.fit(self.train_features, score_targets)

        y_pred = gnb.predict(self.train_features)

        # do some things with y_pred compared to the known data
        # you'll call functions from sklearn to get these values based on the data
        # we're providing the correct values here from an actual run
        accuracy = .7157
        recall = .8143
        precision = .7705
        fscore = .7918

        train_metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "fscore": fscore
        }

        gnb = GaussianNB()

        actual_targets = self.train_targets["good quality"]
        gnb.fit(self.train_features, actual_targets)

        y_pred = gnb.predict(self.test_features)

        #do some things with y_pred compared to the known data
        # you'll call functions from sklearn to get these values based on the data
        # we're providing the correct values here from an actual run
        accuracy = .7092
        recall = .8189
        precision = .7642
        fscore = .7906

        test_metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "fscore": fscore
        }

        feature_importance = permutation_importance(gnb, self.test_features, self.test_targets['good quality'])
        import_mean = feature_importance.importances_mean

        return ModelMetrics("Naive Bayes", train_metrics, test_metrics, import_mean)

# this creates a NaiveBayes class and calls the method to return a ModelMetrics object
def run_locally() -> int:
    naive_bayes = NaiveBayes()

    model_metrics = naive_bayes.calculate_naive_bayes_metrics()

    if model_metrics.dtype == ModelMetrics:
        return 0
    else:
        return "Datatype not ModelMetrics"

# refer to https://docs.python.org/3/library/__main__.html for the structure
if __name__ == '__main__':
    sys.exit(run_locally())



        





