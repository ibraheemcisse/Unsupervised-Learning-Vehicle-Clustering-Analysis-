## Overview

This document describes the methodology used for clustering vehicles based on their physical and performance characteristics using the K-means unsupervised learning algorithm.

## Dataset Description

### Features
- **Weight**: Vehicle weight in pounds (1000-3000 lbs)
- **Engine Size**: Engine displacement in liters (1.0-4.0 L)
- **Horsepower**: Engine horsepower (50-300 HP)

### Data Generation
For demonstration purposes, we generate synthetic data that mimics realistic vehicle characteristics:
- Base features are randomly generated within realistic ranges
- Horsepower is partially correlated with weight and engine size
- Gaussian noise is added for realism

## Preprocessing

### Feature Scaling
All features are standardized using StandardScaler to ensure equal contribution to distance calculations:
```
z = (x - μ) / σ
```
where μ is the mean and σ is the standard deviation.

### Rationale
- Weight (1000-3000) has much larger scale than Engine Size (1.0-4.0)
- Without scaling, weight would dominate distance calculations
- Standardization ensures each feature contributes equally

## K-means Algorithm

### Algorithm Steps
1. Initialize k cluster centroids randomly
2. Assign each point to nearest centroid (Euclidean distance)
3. Update centroids to mean of assigned points
4. Repeat steps 2-3 until convergence

### Distance Metric
Euclidean distance in standardized feature space:
```
d(p,c) = √[(p₁-c₁)² + (p₂-c₂)² + (p₃-c₃)²]
```

## Optimal K Selection

### Methods Used

#### 1. Elbow Method
- Plot within-cluster sum of squares (WCSS) vs k
- Look for "elbow" where rate of decrease slows
- WCSS = Σᵢ Σₓ∈Cᵢ ||x - cᵢ||²

#### 2. Silhouette Analysis
- Measures how similar points are to their cluster vs other clusters
- Silhouette coefficient: s = (b - a) / max(a, b)
- Where a = average distance to points in same cluster
- And b = average distance to points in nearest cluster
- Range: [-1, 1], higher is better

### Typical Results
- Elbow method often suggests k=3 for vehicle data
- Silhouette analysis confirms k=3 as optimal
- k=3 naturally separates into: Economy, Mid-range, Performance

## Cluster Interpretation

### Expected Clusters

#### Cluster 0: Economy Vehicles
- Lower weight (< 1700 lbs)
- Smaller engines (< 2.0 L)
- Lower horsepower (< 150 HP)
- Examples: Compact cars, economy vehicles

#### Cluster 1: Mid-range Vehicles
- Moderate weight (1700-2300 lbs)
- Medium engines (2.0-3.0 L)
- Moderate horsepower (150-220 HP)
- Examples: Sedans, mid-size SUVs

#### Cluster 2: Performance Vehicles
- Higher weight (> 2300 lbs)
- Larger engines (> 3.0 L)
- Higher horsepower (> 220 HP)
- Examples: Sports cars, large SUVs, trucks

## Validation Metrics

### Internal Validation
- **Inertia**: Sum of squared distances to centroids (lower is better)
- **Silhouette Score**: Average silhouette coefficient (higher is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion

### Business Validation
- Do clusters make intuitive sense?
- Are cluster characteristics actionable?
- Do clusters align with known vehicle categories?

## Limitations

### K-means Assumptions
- Clusters are spherical and similar sized
- Features are equally important
- Linear separability in feature space

### Dataset Limitations
- Synthetic data may not capture real-world complexity
- Limited features (missing fuel efficiency, price, etc.)
- No categorical features (brand, type, etc.)

## Applications

### Business Use Cases
1. **Market Segmentation**: Identify distinct vehicle categories
2. **Product Development**: Understand feature relationships
3. **Pricing Strategy**: Group vehicles by performance tier
4. **Inventory Management**: Cluster vehicles by demand patterns

### Technical Extensions
1. **Feature Engineering**: Add derived features (power-to-weight ratio)
2. **Advanced Algorithms**: Try DBSCAN, Gaussian Mixture Models
3. **Dimensionality Reduction**: Use PCA for visualization
4. **Real Data**: Apply to actual vehicle databases

## Conclusion

K-means clustering successfully identifies meaningful vehicle segments based on physical and performance characteristics. The three-cluster solution provides actionable insights for business applications while maintaining interpretability.

Future work should focus on incorporating real vehicle data and additional features to improve cluster quality and business relevance.## Overview

This document describes the methodology used for clustering vehicles based on their physical and performance characteristics using the K-means unsupervised learning algorithm.

## Dataset Description

### Features
- **Weight**: Vehicle weight in pounds (1000-3000 lbs)
- **Engine Size**: Engine displacement in liters (1.0-4.0 L)
- **Horsepower**: Engine horsepower (50-300 HP)

### Data Generation
For demonstration purposes, we generate synthetic data that mimics realistic vehicle characteristics:
- Base features are randomly generated within realistic ranges
- Horsepower is partially correlated with weight and engine size
- Gaussian noise is added for realism

## Preprocessing

### Feature Scaling
All features are standardized using StandardScaler to ensure equal contribution to distance calculations:
```
z = (x - μ) / σ
```
where μ is the mean and σ is the standard deviation.

### Rationale
- Weight (1000-3000) has much larger scale than Engine Size (1.0-4.0)
- Without scaling, weight would dominate distance calculations
- Standardization ensures each feature contributes equally

## K-means Algorithm

### Algorithm Steps
1. Initialize k cluster centroids randomly
2. Assign each point to nearest centroid (Euclidean distance)
3. Update centroids to mean of assigned points
4. Repeat steps 2-3 until convergence

### Distance Metric
Euclidean distance in standardized feature space:
```
d(p,c) = √[(p₁-c₁)² + (p₂-c₂)² + (p₃-c₃)²]
```

## Optimal K Selection

### Methods Used

#### 1. Elbow Method
- Plot within-cluster sum of squares (WCSS) vs k
- Look for "elbow" where rate of decrease slows
- WCSS = Σᵢ Σₓ∈Cᵢ ||x - cᵢ||²

#### 2. Silhouette Analysis
- Measures how similar points are to their cluster vs other clusters
- Silhouette coefficient: s = (b - a) / max(a, b)
- Where a = average distance to points in same cluster
- And b = average distance to points in nearest cluster
- Range: [-1, 1], higher is better

### Typical Results
- Elbow method often suggests k=3 for vehicle data
- Silhouette analysis confirms k=3 as optimal
- k=3 naturally separates into: Economy, Mid-range, Performance

## Cluster Interpretation

### Expected Clusters

#### Cluster 0: Economy Vehicles
- Lower weight (< 1700 lbs)
- Smaller engines (< 2.0 L)
- Lower horsepower (< 150 HP)
- Examples: Compact cars, economy vehicles

#### Cluster 1: Mid-range Vehicles
- Moderate weight (1700-2300 lbs)
- Medium engines (2.0-3.0 L)
- Moderate horsepower (150-220 HP)
- Examples: Sedans, mid-size SUVs

#### Cluster 2: Performance Vehicles
- Higher weight (> 2300 lbs)
- Larger engines (> 3.0 L)
- Higher horsepower (> 220 HP)
- Examples: Sports cars, large SUVs, trucks

## Validation Metrics

### Internal Validation
- **Inertia**: Sum of squared distances to centroids (lower is better)
- **Silhouette Score**: Average silhouette coefficient (higher is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion

### Business Validation
- Do clusters make intuitive sense?
- Are cluster characteristics actionable?
- Do clusters align with known vehicle categories?

## Limitations

### K-means Assumptions
- Clusters are spherical and similar sized
- Features are equally important
- Linear separability in feature space

### Dataset Limitations
- Synthetic data may not capture real-world complexity
- Limited features (missing fuel efficiency, price, etc.)
- No categorical features (brand, type, etc.)

## Applications

### Business Use Cases
1. **Market Segmentation**: Identify distinct vehicle categories
2. **Product Development**: Understand feature relationships
3. **Pricing Strategy**: Group vehicles by performance tier
4. **Inventory Management**: Cluster vehicles by demand patterns

### Technical Extensions
1. **Feature Engineering**: Add derived features (power-to-weight ratio)
2. **Advanced Algorithms**: Try DBSCAN, Gaussian Mixture Models
3. **Dimensionality Reduction**: Use PCA for visualization
4. **Real Data**: Apply to actual vehicle databases

## Conclusion

K-means clustering successfully identifies meaningful vehicle segments based on physical and performance characteristics. The three-cluster solution provides actionable insights for business applications while maintaining interpretability.

Future work should focus on incorporating real vehicle data and additional features to improve cluster quality and business relevance.
