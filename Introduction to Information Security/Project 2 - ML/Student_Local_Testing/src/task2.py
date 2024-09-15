import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection

def tts(  dataset: pd.DataFrame,
                       label_col: str, 
                       test_size: float,
                       stratify: bool,
                       random_state: int) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    
    train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(
        dataset.drop(columns=[label_col]),
        dataset[label_col],
        test_size=test_size,
        stratify=dataset[label_col],
        random_state=random_state
    )
    return train_features,test_features,train_labels,test_labels

class PreprocessDataset:
    def __init__(self, 
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                 ):
        
        self.one_hot_encode_cols = one_hot_encode_cols
        self.min_max_scale_cols = min_max_scale_cols
        self.n_components = n_components
        self.feature_engineering_functions = feature_engineering_functions

        self.encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.train_columns = None


    def one_hot_encode_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        df_encode_cols = train_features[self.one_hot_encode_cols]
        df_other_cols = train_features.drop(columns=self.one_hot_encode_cols)

        encoded = self.encoder.fit_transform(df_encode_cols)
        features = self.encoder.get_feature_names_out(self.one_hot_encode_cols)

        df_encoded = pd.DataFrame(encoded, columns=features, index=train_features.index)
        df_concat = pd.concat([df_encoded, df_other_cols], axis=1)

        self.train_columns = df_encoded.columns

        return df_concat

    def one_hot_encode_columns_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        df_encode_cols = test_features[self.one_hot_encode_cols]
        df_other_cols = test_features.drop(columns=self.one_hot_encode_cols)

        encoded = self.encoder.fit_transform(df_encode_cols)
        features = self.encoder.get_feature_names_out(self.one_hot_encode_cols)

        df_encoded = pd.DataFrame(encoded, columns=self.train_columns, index=test_features.index)
        df_concat = pd.concat([df_encoded, df_other_cols], axis=1)

        return df_concat
        

    def min_max_scaled_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        min_max_scaled_dataset = pd.DataFrame()
        return min_max_scaled_dataset

    def min_max_scaled_columns_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        min_max_scaled_dataset = pd.DataFrame()
        return min_max_scaled_dataset
    
    def pca_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        pca_dataset = pd.DataFrame()
        return pca_dataset

    def pca_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        pca_dataset = pd.DataFrame()
        return pca_dataset

    def feature_engineering_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        feature_engineered_dataset = pd.DataFrame()
        return feature_engineered_dataset
    
    def feature_engineering_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        feature_engineered_dataset = pd.DataFrame()
        return feature_engineered_dataset

    def preprocess_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        preprocessed_dataset = pd.DataFrame()
        return preprocessed_dataset
    
    def preprocess_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        preprocessed_dataset = pd.DataFrame()
        return preprocessed_dataset