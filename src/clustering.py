import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

class VehicleClustering:
    """
    Vehicle clustering analysis using K-means algorithm
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.data = None
        self.scaled_data = None
        self.labels = None
        self.centroids = None
        
    def fit(self, data):
        """
        Fit the clustering model to the data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Vehicle data with features: Weight, EngineSize, Horsepower
        """
        self.data = data.copy()
        
        # Scale the data
        feature_cols = ['Weight', 'EngineSize', 'Horsepower']
        self.scaled_data = self.scaler.fit_transform(data[feature_cols])
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                           random_state=self.random_state, 
                           n_init=10)
        self.labels = self.kmeans.fit_predict(self.scaled_data)
        
        # Store centroids in original scale
        self.centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        # Add cluster labels to data
        self.data['Cluster'] = self.labels
        
        return self
    
    def find_optimal_k(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        
        Parameters:
        -----------
        k_range : range, default=range(2, 11)
            Range of k values to test
            
        Returns:
        --------
        dict : Results containing inertias and silhouette scores
        """
        if self.scaled_data is None:
            raise ValueError("Must fit data first or call with data")
            
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels_temp = kmeans_temp.fit_predict(self.scaled_data)
            
            inertias.append(kmeans_temp.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, labels_temp))
        
        return {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_silhouette': k_range[np.argmax(silhouette_scores)]
        }
    
    def plot_clusters(self, feature_pairs=None, figsize=(15, 10)):
        """
        Create comprehensive cluster visualization
        
        Parameters:
        -----------
        feature_pairs : list of tuples, optional
            Feature pairs to plot. Default uses all combinations
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if self.data is None or 'Cluster' not in self.data.columns:
            raise ValueError("Must fit clustering model first")
        
        if feature_pairs is None:
            feature_pairs = [('Weight', 'Horsepower'), 
                           ('Weight', 'EngineSize'), 
                           ('EngineSize', 'Horsepower')]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_clusters))
        
        # Scatter plots
        for i, (x_feature, y_feature) in enumerate(feature_pairs):
            ax = axes[0, i]
            
            for cluster in range(self.n_clusters):
                cluster_data = self.data[self.data['Cluster'] == cluster]
                ax.scatter(cluster_data[x_feature], cluster_data[y_feature], 
                          c=[colors[cluster]], label=f'Cluster {cluster}', 
                          alpha=0.7, s=50)
            
            # Plot centroids
            centroids_df = pd.DataFrame(self.centroids, 
                                      columns=['Weight', 'EngineSize', 'Horsepower'])
            ax.scatter(centroids_df[x_feature], centroids_df[y_feature], 
                      c='black', marker='x', s=200, linewidths=3, label='Centroids')
            
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f'{x_feature} vs {y_feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Cluster statistics
        axes[1, 0].pie(self.data['Cluster'].value_counts().sort_index(), 
                       labels=[f'Cluster {i}' for i in range(self.n_clusters)],
                       colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Cluster Size Distribution')
        
        # Centroid comparison
        centroids_df = pd.DataFrame(self.centroids, 
                                  columns=['Weight', 'EngineSize', 'Horsepower'],
                                  index=[f'Cluster {i}' for i in range(self.n_clusters)])
        centroids_df.plot(kind='bar', ax=axes[1, 1], color=['skyblue', 'lightgreen', 'salmon'])
        axes[1, 1].set_title('Cluster Centroids')
        axes[1, 1].set_ylabel('Feature Values')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Feature distributions by cluster
        df_melted = self.data.melt(id_vars=['Cluster'], 
                                  value_vars=['Weight', 'EngineSize', 'Horsepower'],
                                  var_name='Feature', value_name='Value')
        sns.boxplot(data=df_melted, x='Feature', y='Value', hue='Cluster', ax=axes[1, 2])
        axes[1, 2].set_title('Feature Distribution by Cluster')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_cluster_summary(self):
        """
        Get detailed cluster statistics and interpretation
        
        Returns:
        --------
        pd.DataFrame : Cluster summary statistics
        """
        if self.data is None or 'Cluster' not in self.data.columns:
            raise ValueError("Must fit clustering model first")
        
        summary = self.data.groupby('Cluster').agg({
            'Weight': ['mean', 'std', 'min', 'max', 'count'],
            'EngineSize': ['mean', 'std', 'min', 'max'],
            'Horsepower': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        return summary
    
    def predict(self, new_data):
        """
        Predict cluster for new vehicle data
        
        Parameters:
        -----------
        new_data : pd.DataFrame
            New vehicle data to classify
            
        Returns:
        --------
        np.array : Predicted cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Must fit clustering model first")
        
        scaled_new_data = self.scaler.transform(new_data[['Weight', 'EngineSize', 'Horsepower']])
        return self.kmeans.predict(scaled_new_data)
