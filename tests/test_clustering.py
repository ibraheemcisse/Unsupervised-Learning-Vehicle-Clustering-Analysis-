import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_vehicle_data
from src.clustering import VehicleClustering

class TestVehicleClustering:
    
    def setup_method(self):
        """Set up test data"""
        self.data = generate_vehicle_data(n_samples=100, random_state=42)
        self.clusterer = VehicleClustering(n_clusters=3, random_state=42)
    
    def test_data_generation(self):
        """Test data generation"""
        assert len(self.data) == 100
        assert list(self.data.columns) == ['Weight', 'EngineSize', 'Horsepower']
        assert self.data['Weight'].min() >= 1000
        assert self.data['Weight'].max() <= 3000
        assert self.data['EngineSize'].min() >= 1.0
        assert self.data['EngineSize'].max() <= 4.0
        assert self.data['Horsepower'].min() >= 50
        assert self.data['Horsepower'].max() <= 300
    
    def test_clustering_fit(self):
        """Test clustering fit method"""
        self.clusterer.fit(self.data)
        
        assert self.clusterer.labels is not None
        assert len(self.clusterer.labels) == len(self.data)
        assert self.clusterer.centroids is not None
        assert self.clusterer.centroids.shape == (3, 3)
        assert 'Cluster' in self.clusterer.data.columns
    
    def test_optimal_k_finding(self):
        """Test optimal k finding"""
        self.clusterer.fit(self.data)
        results = self.clusterer.find_optimal_k(k_range=range(2, 6))
        
        assert 'k_values' in results
        assert 'inertias' in results
        assert 'silhouette_scores' in results
        assert 'optimal_k_silhouette' in results
        assert len(results['k_values']) == 4
        assert len(results['inertias']) == 4
        assert len(results['silhouette_scores']) == 4
    
    def test_prediction(self):
        """Test prediction on new data"""
        self.clusterer.fit(self.data)
        
        new_data = pd.DataFrame({
            'Weight': [1500, 2500],
            'EngineSize': [2.0, 3.5],
            'Horsepower': [120, 250]
        })
        
        predictions = self.clusterer.predict(new_data)
        assert len(predictions) == 2
        assert all(0 <= pred < 3 for pred in predictions)
    
    def test_cluster_summary(self):
        """Test cluster summary generation"""
        self.clusterer.fit(self.data)
        summary = self.clusterer.get_cluster_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3  # 3 clusters
        assert 'Weight' in summary.columns.get_level_0()
        assert 'EngineSize' in summary.columns.get_level_0()
        assert 'Horsepower' in summary.columns.get_level_0()
