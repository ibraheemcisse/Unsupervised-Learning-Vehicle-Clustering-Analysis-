"""
Vehicle Clustering Analysis Package

This package provides tools for analyzing vehicle data using unsupervised
machine learning techniques, specifically K-means clustering.

Modules:
--------
- data_generator: Generate synthetic vehicle datasets
- clustering: K-means clustering implementation and analysis
- visualization: Plotting and visualization utilities
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_generator import generate_vehicle_data, load_sample_data
from .clustering import VehicleClustering
from .visualization import create_interactive_cluster_plot, plot_elbow_analysis

__all__ = [
    'generate_vehicle_data',
    'load_sample_data', 
    'VehicleClustering',
    'create_interactive_cluster_plot',
    'plot_elbow_analysis'
]
