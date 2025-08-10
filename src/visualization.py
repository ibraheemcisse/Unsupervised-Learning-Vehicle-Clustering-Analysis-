import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_interactive_cluster_plot(data, feature_x='Weight', feature_y='Horsepower'):
    """
    Create interactive cluster visualization using Plotly
    
    Parameters:
    -----------
    data : pd.DataFrame
        Clustered vehicle data
    feature_x : str, default='Weight'
        X-axis feature
    feature_y : str, default='Horsepower'
        Y-axis feature
        
    Returns:
    --------
    plotly.graph_objects.Figure : Interactive plot
    """
    fig = px.scatter(data, x=feature_x, y=feature_y, color='Cluster',
                     title=f'Vehicle Clusters: {feature_x} vs {feature_y}',
                     hover_data=['Weight', 'EngineSize', 'Horsepower'])
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig

def plot_elbow_analysis(k_values, inertias, silhouette_scores):
    """
    Plot elbow method and silhouette analysis results
    
    Parameters:
    -----------
    k_values : list
        K values tested
    inertias : list
        Inertia values for each k
    silhouette_scores : list
        Silhouette scores for each k
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow plot
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_cluster_dashboard(data, centroids):
    """
    Create a comprehensive cluster analysis dashboard
    
    Parameters:
    -----------
    data : pd.DataFrame
        Clustered vehicle data
    centroids : np.array
        Cluster centroids
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Weight vs Horsepower', 'Cluster Distribution', 
                       'Feature Comparison', 'Centroid Analysis'),
        specs=[[{'type': 'scatter'}, {'type': 'pie'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Scatter plot
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        fig.add_trace(
            go.Scatter(x=cluster_data['Weight'], y=cluster_data['Horsepower'],
                      mode='markers', name=f'Cluster {cluster}'),
            row=1, col=1
        )
    
    # Pie chart
    cluster_counts = data['Cluster'].value_counts().sort_index()
    fig.add_trace(
        go.Pie(labels=[f'Cluster {i}' for i in cluster_counts.index],
               values=cluster_counts.values),
        row=1, col=2
    )
    
    # Feature comparison
    feature_means = data.groupby('Cluster')[['Weight', 'EngineSize', 'Horsepower']].mean()
    for feature in ['Weight', 'EngineSize', 'Horsepower']:
        fig.add_trace(
            go.Bar(x=feature_means.index, y=feature_means[feature], name=feature),
            row=2, col=1
        )
    
    # Centroid analysis
    centroids_df = pd.DataFrame(centroids, columns=['Weight', 'EngineSize', 'Horsepower'])
    for i, feature in enumerate(['Weight', 'EngineSize', 'Horsepower']):
        fig.add_trace(
            go.Bar(x=[f'Cluster {j}' for j in range(len(centroids))], 
                   y=centroids_df[feature], name=f'{feature} Centroid'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Vehicle Clustering Dashboard")
    
    return fig
