import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_vehicle_data, load_sample_data

class TestDataGenerator:
    
    def test_generate_vehicle_data_default(self):
        """Test default data generation"""
        data = generate_vehicle_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 300
        assert list(data.columns) == ['Weight', 'EngineSize', 'Horsepower']
    
    def test_generate_vehicle_data_custom(self):
        """Test custom data generation"""
        data = generate_vehicle_data(n_samples=150, random_state=123)
        
        assert len(data) == 150
        assert data['Weight'].dtype in [np.int32, np.int64]
        assert data['EngineSize'].dtype in [np.float32, np.float64]
        assert data['Horsepower'].dtype in [np.int32, np.int64]
    
    def test_data_bounds(self):
        """Test data is within expected bounds"""
        data = generate_vehicle_data(n_samples=50)
        
        assert data['Weight'].min() >= 1000
        assert data['Weight'].max() <= 3000
        assert data['EngineSize'].min() >= 1.0
        assert data['EngineSize'].max() <= 4.0
        assert data['Horsepower'].min() >= 50
        assert data['Horsepower'].max() <= 300
    
    def test_reproducibility(self):
        """Test data generation is reproducible"""
        data1 = generate_vehicle_data(n_samples=100, random_state=42)
        data2 = generate_vehicle_data(n_samples=100, random_state=42)
        
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_load_sample_data(self):
        """Test sample data loading"""
        data = load_sample_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 300
        assert list(data.columns) == ['Weight', 'EngineSize', 'Horsepower']
