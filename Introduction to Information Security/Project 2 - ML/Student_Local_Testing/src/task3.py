import numpy as np
import pandas as pd
import sklearn.cluster
import yellowbrick.cluster

class KmeansClustering:
    def __init__(self, 
                 random_state: int
                ):
        
        self.random_state = random_state
        self.k = None
        self.model = None

    def kmeans_train(self,train_features:pd.DataFrame) -> list:
        n_init = 10
        initial = sklearn.cluster.KMeans(random_state=self.random_state, n_init=n_init)
        
        elbow = yellowbrick.cluster.KElbowVisualizer(initial, k=(1, 10))
        elbow.fit(train_features)
        self.k = elbow.elbow_value_

        self.model = sklearn.cluster.KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)        
        self.model.fit(train_features)

        return self.model.labels_.tolist()

    def kmeans_test(self,test_features:pd.DataFrame) -> list:
        return self.model.predict(test_features).tolist()

    def train_add_kmeans_cluster_id_feature(self,train_features:pd.DataFrame) -> pd.DataFrame:
        cluster_ids = self.kmeans_train(train_features)
        train_features['kmeans_cluster_id'] = cluster_ids
        return train_features

    def test_add_kmeans_cluster_id_feature(self,test_features:pd.DataFrame) -> pd.DataFrame:
        cluster_ids = self.kmeans_test(test_features)
        test_features['kmeans_cluster_id'] = cluster_ids
        return test_features