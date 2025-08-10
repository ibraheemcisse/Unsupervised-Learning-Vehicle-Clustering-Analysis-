# Unsupervised-Learning-Vehicle-Clustering-Analysis-
A comprehensive machine learning project demonstrating unsupervised learning through K-means clustering on vehicle data.

## 📁 Repository Structure

```
vehicle-clustering/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── vehicle_data.csv
│   └── processed/
│       └── clustered_data.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_kmeans_clustering.ipynb
│   └── 03_advanced_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── clustering.py
│   └── visualization.py
├── web_app/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── images/
│   └── cluster_plots/
└── docs/
    └── methodology.md
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/vehicle-clustering.git
cd vehicle-clustering
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

### 4. Launch Interactive Web App
Open `web_app/index.html` in your browser or serve it locally:
```bash
cd web_app
python -m http.server 8000
# Visit http://localhost:8000
```

## 📊 Project Overview

This project demonstrates **unsupervised machine learning** using K-means clustering to segment vehicles based on their characteristics:

- **Weight** (1000-3000 lbs)
- **Engine Size** (1.0-4.0 L)
- **Horsepower** (50-300 HP)

### Key Features
- Interactive web visualization
- Jupyter notebook analysis
- Modular Python code
- Comprehensive documentation

## 🔧 Technologies Used

- **Python**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Jupyter**: Interactive analysis and visualization
- **HTML/CSS/JavaScript**: Web-based interactive tool
- **Plotly.js**: Advanced interactive plots

## 📈 Analysis Highlights

### Clustering Results
- **Cluster 1**: Economy vehicles (low weight, small engines)
- **Cluster 2**: Mid-range vehicles (moderate specs)
- **Cluster 3**: Performance vehicles (high horsepower)

### Methodologies
1. **Data Generation**: Synthetic vehicle dataset
2. **Preprocessing**: Feature scaling and normalization
3. **Clustering**: K-means with elbow method for optimal K
4. **Validation**: Silhouette analysis and inertia plots

## 🎯 Usage Examples

### Python/Jupyter Usage
```python
from src.clustering import VehicleClustering
from src.data_generator import generate_vehicle_data

# Generate data
data = generate_vehicle_data(n_samples=300)

# Perform clustering
clusterer = VehicleClustering(n_clusters=3)
clusterer.fit(data)

# Visualize results
clusterer.plot_clusters()
```

### Web App Features
- **Real-time clustering**: Adjust parameters instantly
- **Multiple views**: Different feature combinations
- **Statistics dashboard**: Cluster characteristics
- **Export functionality**: Save results and plots

# 🚗 Complete Setup and Usage Guide

## 📋 Table of Contents
1. [GitHub Repository Setup](#github-repository-setup)
2. [Interactive Web UI Usage](#interactive-web-ui-usage)
3. [Jupyter Notebook Setup](#jupyter-notebook-setup)
4. [Python Module Usage](#python-module-usage)
5. [Advanced Features](#advanced-features)

---

## 🔧 GitHub Repository Setup

### Step 1: Create GitHub Repository
1. Go to [GitHub](https://github.com) and create a new repository
2. Name it `vehicle-clustering`
3. Add description: "Vehicle clustering analysis using machine learning"
4. Initialize with README ✅

### Step 2: Clone and Setup Local Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/vehicle-clustering.git
cd vehicle-clustering

# Create directory structure
mkdir -p {data/{raw,processed},notebooks,src,web_app,images/cluster_plots,docs,tests}

# Create files (copy content from artifacts above)
touch requirements.txt .gitignore setup.py LICENSE
touch src/{__init__.py,data_generator.py,clustering.py,visualization.py}
touch notebooks/{01_data_exploration.ipynb,02_kmeans_clustering.ipynb,03_advanced_analysis.ipynb}
touch tests/{__init__.py,test_clustering.py,test_data_generator.py}
touch docs/methodology.md
touch web_app/{index.html,style.css,script.js}
```

### Step 3: Add Content to Files
Copy the content from the artifacts I created above into the respective files.

### Step 4: Initial Commit
```bash
# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Vehicle clustering analysis project

- Add Python modules for data generation and clustering
- Add Jupyter notebooks for analysis
- Add interactive web application
- Add comprehensive documentation
- Add test suite and CI/CD pipeline"

# Push to GitHub
git push origin main
```

---

## 🌐 Interactive Web UI Usage

### Setup
1. **Download the HTML file** from the web_app artifact above
2. **Save as `index.html`** in your web_app folder
3. **Open in browser** or serve locally:

```bash
# Option 1: Direct file opening
# Simply double-click index.html or open in browser

# Option 2: Local server (recommended)
cd web_app
python -m http.server 8000
# Visit: http://localhost:8000
```

### Features and Usage

#### 🎛️ Control Panel
- **Number of Clusters**: Change from 2-5 to see different groupings
- **Dataset Size**: Adjust from 100-1000 vehicles
- **Axis Selection**: Choose which features to plot (X vs Y)
- **Generate New Data**: Create fresh random dataset
- **Run Clustering**: Apply K-means with current settings

#### 📊 Visualizations
1. **Scatter Plot**: Main clustering visualization
   - Each color represents a different cluster
   - Black X marks show cluster centroids
   - Hover for detailed information

2. **Cluster Centers**: Bar chart showing average feature values
   - Compare characteristics across clusters
   - Understand what makes each cluster unique

3. **Statistics Cards**: Real-time cluster metrics
   - Number of vehicles per cluster
   - Average weight, engine size, and horsepower

#### 🎯 Interactive Exploration
```javascript
// Example: Explore different scenarios
1. Start with default settings (3 clusters, 300 vehicles)
2. Try k=2 to see basic economy vs performance split
3. Try k=5 for more granular segmentation
4. Change axes to "EngineSize vs Horsepower" for different perspective
5. Generate new data to see algorithm consistency
```

---

## 📊 Jupyter Notebook Setup

### Installation
```bash
# Install required packages
pip install -r requirements.txt

# Install Jupyter (if not already installed)
pip install jupyter notebook ipywidgets

# Enable widgets (for interactive plots)
jupyter nbextension enable --py widgetsnbextension
```

### Starting Jupyter
```bash
# Navigate to project directory
cd vehicle-clustering

# Start Jupyter
jupyter notebook

# This opens browser at: http://localhost:8888
```

### Notebook Usage Guide

#### 📓 01_data_exploration.ipynb
```python
# What you'll learn:
- Data generation and characteristics
- Statistical summaries and distributions
- Feature relationships and correlations
- Data visualization techniques

# Key sections:
1. Load and inspect data
2. Statistical analysis
3. Correlation analysis
4. Distribution plots
5. Feature relationships
```

#### 📓 02_kmeans_clustering.ipynb (Main Analysis)
```python
# Complete clustering workflow:
1. Data preprocessing and scaling
2. Optimal k selection (elbow method + silhouette)
3. K-means clustering execution
4. Results visualization and interpretation
5. Cluster statistics and business insights

# Code example:
from src.clustering import VehicleClustering
from src.data_generator import generate_vehicle_data

# Generate data
data = generate_vehicle_data(n_samples=300)

# Perform clustering
clusterer = VehicleClustering(n_clusters=3)
clusterer.fit(data)

# Visualize results
clusterer.plot_clusters()

# Get detailed statistics
summary = clusterer.get_cluster_summary()
print(summary)
```

#### 📓 03_advanced_analysis.ipynb
```python
# Advanced techniques:
- PCA for dimensionality reduction
- Different clustering algorithms comparison
- Parameter sensitivity analysis
- Custom visualization techniques
```

### Interactive Features in Jupyter
```python
# Use ipywidgets for interactive exploration
from ipywidgets import interact, IntSlider, Dropdown

@interact(
    n_clusters=IntSlider(min=2, max=8, value=3),
    dataset_size=Dropdown(options=[100, 300, 500, 1000], value=300)
)
def interactive_clustering(n_clusters, dataset_size):
    data = generate_vehicle_data(n_samples=dataset_size)
    clusterer = VehicleClustering(n_clusters=n_clusters)
    clusterer.fit(data)
    clusterer.plot_clusters()
    return clusterer.get_cluster_summary()
```

---

## 🐍 Python Module Usage

### Basic Usage
```python
# Import modules
from src.data_generator import generate_vehicle_data
from src.clustering import VehicleClustering
from src.visualization import create_interactive_cluster_plot

# Generate data
data = generate_vehicle_data(n_samples=500, random_state=42)

# Initialize and fit clusterer
clusterer = VehicleClustering(n_clusters=3, random_state=42)
clusterer.fit(data)

# Get results
labels = clusterer.labels
centroids = clusterer.centroids
clustered_data = clusterer.data  # Original data + cluster labels
```

### Advanced Features
```python
# Find optimal number of clusters
results = clusterer.find_optimal_k(k_range=range(2, 10))
optimal_k = results['optimal_k_silhouette']

# Refit with optimal k
clusterer = VehicleClustering(n_clusters=optimal_k)
clusterer.fit(data)

# Predict new vehicles
new_vehicles = pd.DataFrame({
    'Weight': [1500, 2800, 2200],
    'EngineSize': [1.8, 3.5, 2.4],
    'Horsepower': [110, 280, 180]
})
predictions = clusterer.predict(new_vehicles)
print(f"New vehicles belong to clusters: {predictions}")

# Get detailed analysis
summary = clusterer.get_cluster_summary()
```

### Custom Visualization
```python
# Create interactive plots
from src.visualization import create_interactive_cluster_plot

fig = create_interactive_cluster_plot(
    clusterer.data, 
    feature_x='Weight', 
    feature_y='Horsepower'
)
fig.show()

# Save plots
import matplotlib.pyplot as plt
clusterer.plot_clusters()
plt.savefig('images/cluster_plots/vehicle_clusters.png', dpi=300, bbox_inches='tight')
```

---

## 🚀 Advanced Features

### Command Line Interface
```bash
# Install package in development mode
pip install -e .

# Use command line tool
vehicle-cluster --data-size 500 --clusters 4 --output results.csv
```

### Integration with Real Data
```python
# Load real vehicle data
import pandas as pd

real_data = pd.read_csv('data/raw/real_vehicle_data.csv')
# Ensure columns: ['Weight', 'EngineSize', 'Horsepower']

# Apply clustering
clusterer = VehicleClustering(n_clusters=3)
clusterer.fit(real_data)

# Export results
clusterer.data.to_csv('data/processed/real_vehicles_clustered.csv', index=False)
```

### API Integration
```python
# Create simple API endpoint
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/cluster', methods=['POST'])
def cluster_vehicles():
    data = request.json
    df = pd.DataFrame(data)
    
    clusterer = VehicleClustering(n_clusters=3)
    clusterer.fit(df)
    
    return jsonify({
        'clusters': clusterer.labels.tolist(),
        'centroids': clusterer.centroids.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### Performance Optimization
```python
# For large datasets
from sklearn.cluster import MiniBatchKMeans

class OptimizedVehicleClustering(VehicleClustering):
    def __init__(self, n_clusters=3, random_state=42, use_mini_batch=False):
        super().__init__(n_clusters, random_state)
        self.use_mini_batch = use_mini_batch
        
    def fit(self, data):
        if self.use_mini_batch and len(data) > 1000:
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=100
            )
        return super().fit(data)

# Usage for large datasets
large_data = generate_vehicle_data(n_samples=10000)
fast_clusterer = OptimizedVehicleClustering(use_mini_batch=True)
fast_clusterer.fit(large_data)
```

---

## 🎓 Learning Path

### Beginner (Start Here)
1. **Open the Interactive Web UI** - Get familiar with clustering concepts
2. **Run the main Jupyter notebook** - `02_kmeans_clustering.ipynb`
3. **Experiment with different parameters** - Change k, dataset size
4. **Understand the results** - What do the clusters represent?

### Intermediate
1. **Modify the Python modules** - Add new features or algorithms
2. **Create custom visualizations** - Use the visualization module
3. **Work with real data** - Replace synthetic data with actual vehicle data
4. **Implement additional clustering algorithms** - DBSCAN, Gaussian Mixture

### Advanced
1. **Build a web application** - Create a full dashboard
2. **Add machine learning pipeline** - Feature engineering, model selection
3. **Deploy to cloud** - AWS, Azure, or Google Cloud
4. **Create API service** - RESTful API for clustering service

---

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# If module not found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or install in development mode
pip install -e .
```

#### Jupyter Kernel Issues
```bash
# Install IPython kernel
python -m ipykernel install --user --name vehicle-clustering
```

#### Package Dependencies
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate vehicle-clustering
```

#### Web App Not Loading
```bash
# Check browser console for errors
# Ensure JavaScript is enabled
# Try different browser or incognito mode
```

### Getting Help
- 📖 Read the methodology.md documentation
- 🐛 Check the test suite: `pytest tests/`
- 💬 Open GitHub issues for bugs
- 📧 Contact maintainers for questions

---

## 🎉 Next Steps

1. **Star the repository** ⭐ if you find it useful
2. **Fork and experiment** 🍴 with your own modifications
3. **Contribute improvements** 🤝 via pull requests
4. **Share your results** 📱 on social media
5. **Apply to real projects** 🚀 in your work or research

Happy clustering! 🚗💨

## 📚 Learning Objectives

This project teaches:
- **Unsupervised Learning**: Finding patterns without labels
- **K-means Algorithm**: Centroid-based clustering
- **Feature Engineering**: Data preprocessing techniques
- **Visualization**: Multiple plotting approaches
- **Web Development**: Interactive data applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Tags

`machine-learning` `unsupervised-learning` `kmeans` `clustering` `python` `jupyter` `data-science` `visualization` `interactive`

---

**Author**: Your Name  
**Contact**: your.email@example.com  
**Project Link**: https://github.com/yourusername/vehicle-clustering
