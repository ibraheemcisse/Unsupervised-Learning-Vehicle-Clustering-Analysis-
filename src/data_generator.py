import pandas as pd
import numpy as np

def generate_vehicle_data(n_samples=300, random_state=42, noise_level=0.1):
    """
    Generate synthetic vehicle dataset with realistic relationships
    
    Parameters:
    -----------
    n_samples : int, default=300
        Number of vehicle samples to generate
    random_state : int, default=42
        Random seed for reproducibility
    noise_level : float, default=0.1
        Amount of noise to add to make data more realistic
    
    Returns:
    --------
    pd.DataFrame : Generated vehicle data
    """
    np.random.seed(random_state)
    
    # Generate base features
    weight = np.random.randint(1000, 3000, n_samples)
    engine_size = np.random.uniform(1.0, 4.0, n_samples)
    
    # Create realistic horsepower based on weight and engine size
    # Larger engines and heavier cars tend to have more horsepower
    base_hp = (weight / 20) + (engine_size * 40) + np.random.normal(0, 30, n_samples)
    horsepower = np.clip(base_hp, 50, 300).astype(int)
    
    # Add some noise for realism
    if noise_level > 0:
        weight += np.random.normal(0, weight * noise_level, n_samples).astype(int)
        engine_size += np.random.normal(0, engine_size * noise_level, n_samples)
        horsepower += np.random.normal(0, horsepower * noise_level, n_samples).astype(int)
    
    # Ensure realistic bounds
    weight = np.clip(weight, 1000, 3000)
    engine_size = np.clip(engine_size, 1.0, 4.0)
    horsepower = np.clip(horsepower, 50, 300)
    
    data = {
        'Weight': weight,
        'EngineSize': engine_size,
        'Horsepower': horsepower
    }
    
    return pd.DataFrame(data)

def load_sample_data():
    """Load sample dataset for testing"""
    return generate_vehicle_data(n_samples=300)
