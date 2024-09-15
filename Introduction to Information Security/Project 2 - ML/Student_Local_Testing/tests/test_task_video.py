import unittest
import pandas as pd
import os
import sys
import pickle

if os.path.split(os.getcwd())[-1]=="Student_Local_Testing":
    folder_loc="main"
    sys.path.append(os.getcwd())
    from src.task_video import (NaiveBayes, ModelMetrics)
elif os.path.split(os.getcwd())[-1]=="tests":
    folder_loc="tests"
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
    from src.task_video import (NaiveBayes, ModelMetrics)
else:
    raise Exception(f"Running Tests from `{os.path.split(os.getcwd())[-1]}`. Please run the tests with your CWD set to either Student_Local_Testing or tests folders")

class Test_calculate_Naive_Bayes(unittest.TestCase):
    def setUp(self):
        # Train and Test Data is derived from http://archive.ics.uci.edu/dataset/186/wine+quality
        if folder_loc=="main":
            folder_path4 = os.path.join(os.getcwd(),"task4")
        else:
            folder_path4 = os.path.join(os.getcwd(),"..","task4")
        pkl_files_folder = os.path.join(folder_path4,"pkl_files")
        # Inputs
        self.train_features = pd.read_csv(os.path.join(folder_path4,"wine_train_features.csv"), index_col=0)
        self.test_features  = pd.read_csv(os.path.join(folder_path4,"wine_test_features.csv"), index_col=0)
        self.train_targets  = pd.read_csv(os.path.join(folder_path4,"wine_train_targets.csv"), index_col=0)
        self.test_targets   = pd.read_csv(os.path.join(folder_path4,"wine_test_targets.csv"), index_col=0)

        # Answers for Wine Dataset
        with open(os.path.join(pkl_files_folder,"naive_bayes_metrics.pkl"), 'rb') as file:
            self.naive_bayes_metrics_ans = pickle.load(file)

        self.naive_bayes = NaiveBayes()

        # Calculate Metrics with Student's Function
        self.naive_bayes_metrics = self.naive_bayes.calculate_naive_bayes_metrics()

    def test_naive_bayes_accuracy(self):
        naive_bayes_accuracy = round(self.naive_bayes_metrics.train_metrics["accuracy"], 4)
        naive_bayes_accuracy_ans = self.naive_bayes_metrics_ans.train_metrics["accuracy"]
        self.assertEqual(naive_bayes_accuracy_ans,naive_bayes_accuracy)

    def test_naive_bayes_recall(self):
        naive_bayes_recall = round(self.naive_bayes_metrics.train_metrics["recall"], 4)
        naive_bayes_accuracy_ans = self.naive_bayes_metrics_ans.train_metrics["recall"]
        self.assertEqual(naive_bayes_accuracy_ans,naive_bayes_recall)

    def test_naive_bayes_precision(self):
        naive_bayes_precision = round(self.naive_bayes_metrics.train_metrics["precision"], 4)
        naive_bayes_accuracy_ans = self.naive_bayes_metrics_ans.train_metrics["precision"]
        self.assertEqual(naive_bayes_accuracy_ans, naive_bayes_precision)

    def test_naive_bayes_fscore(self):
        naive_bayes_fscore = round(self.naive_bayes_metrics.train_metrics["fscore"], 4)
        naive_bayes_accuracy_ans = self.naive_bayes_metrics_ans.train_metrics["fscore"]
        self.assertEqual(naive_bayes_accuracy_ans, naive_bayes_fscore)

    def test_test_naive_bayes_accuracy(self):
        naive_bayes_accuracy = round(self.naive_bayes_metrics.test_metrics["accuracy"], 4)
        naive_bayes_accuracy_ans = self.naive_bayes_metrics_ans.test_metrics["accuracy"]
        self.assertEqual(naive_bayes_accuracy_ans, naive_bayes_accuracy)

    def test_test_naive_bayes_recall(self):
        naive_bayes_recall = round(self.naive_bayes_metrics.test_metrics["recall"], 4)
        naive_bayes_accuracy_ans = self.naive_bayes_metrics_ans.test_metrics["recall"]
        self.assertEqual(naive_bayes_accuracy_ans, naive_bayes_recall)

    def test_test_naive_bayes_precision(self):
        naive_bayes_precision = round(self.naive_bayes_metrics.test_metrics["precision"], 4)
        test_naive_bayes_precision_ans = self.naive_bayes_metrics_ans.test_metrics["precision"]
        self.assertEqual(test_naive_bayes_precision_ans, naive_bayes_precision)

    def test_test_naive_bayes_fscore(self):
        naive_bayes_fscore = round(self.naive_bayes_metrics.test_metrics["fscore"], 4)
        test_naive_bayes_fscore_ans = self.naive_bayes_metrics_ans.test_metrics["fscore"]
        self.assertEqual(test_naive_bayes_fscore_ans, naive_bayes_fscore)
