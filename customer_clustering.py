import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.data = None
        self.scaled_data = None
        
    def generate_sample_data(self, n_customers=1000):
        """Generate sample customer purchase data"""
        np.random.seed(42)
        
        # Create different customer segments with distinct patterns
        segments = []
        
        # High-value customers (20%)
        n_high = int(n_customers * 0.2)
        high_value = {
            'annual_spending': np.random.normal(5000, 1000, n_high),
            'frequency': np.random.normal(50, 10, n_high),
            'recency_days': np.random.normal(15, 5, n_high),
            'avg_order_value': np.random.normal(200, 50, n_high)
        }
        segments.append(pd.DataFrame(high_value))
        
        # Medium-value customers (50%)
        n_medium = int(n_customers * 0.5)
        medium_value = {
            'annual_spending': np.random.normal(2000, 500, n_medium),
            'frequency': np.random.normal(25, 8, n_medium),
            'recency_days': np.random.normal(30, 10, n_medium),
            'avg_order_value': np.random.normal(80, 20, n_medium)
        }
        segments.append(pd.DataFrame(medium_value))
        
        # Low-value customers (30%)
        n_low = n_customers - n_high - n_medium
        low_value = {
            'annual_spending': np.random.normal(500, 200, n_low),
            'frequency': np.random.normal(8, 3, n_low),
            'recency_days': np.random.normal(60, 20, n_low),
            'avg_order_value': np.random.normal(30, 10, n_low)
        }
        segments.append(pd.DataFrame(low_value))
        
        # Combine all segments
        self.data = pd.concat(segments, ignore_index=True)
        
        # Ensure positive values
        self.data = self.data.abs()
        self.data['customer_id'] = range(1, len(self.data) + 1)
        
        print(f"Generated {len(self.data)} customer records")
        return self.data
    
    def load_data(self, filepath):
        """Load customer data from CSV file"""
        self.data = pd.read_csv(filepath)
        return self.data
    
    def preprocess_data(self):
        """Preprocess and scale the data"""
        # Select features for clustering
        features = ['annual_spending', 'frequency', 'recency_days', 'avg_order_value']
        X = self.data[features]
        
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(X)
        
        print("Data preprocessed and scaled")
        return self.scaled_data
    
    def find_optimal_clusters(self, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, kmeans.labels_))
        
        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow method
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k
    
    def perform_clustering(self, n_clusters=None):
        """Perform K-means clustering"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(self.scaled_data)
        
        # Add cluster labels to original data
        self.data['cluster'] = cluster_labels
        
        print(f"Clustering completed with {n_clusters} clusters")
        return cluster_labels
    
    def analyze_clusters(self):
        """Analyze and describe each cluster"""
        features = ['annual_spending', 'frequency', 'recency_days', 'avg_order_value']
        
        cluster_analysis = self.data.groupby('cluster')[features].agg(['mean', 'std', 'count'])
        
        print("\n=== CLUSTER ANALYSIS ===")
        print(cluster_analysis)
        
        # Create cluster profiles
        profiles = {}
        for cluster_id in sorted(self.data['cluster'].unique()):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'avg_spending': cluster_data['annual_spending'].mean(),
                'avg_frequency': cluster_data['frequency'].mean(),
                'avg_recency': cluster_data['recency_days'].mean(),
                'avg_order_value': cluster_data['avg_order_value'].mean()
            }
            profiles[cluster_id] = profile
            
            print(f"\n--- Cluster {cluster_id} ---")
            print(f"Size: {profile['size']} customers ({profile['size']/len(self.data)*100:.1f}%)")
            print(f"Average Annual Spending: ${profile['avg_spending']:.2f}")
            print(f"Average Purchase Frequency: {profile['avg_frequency']:.1f} times/year")
            print(f"Average Recency: {profile['avg_recency']:.1f} days since last purchase")
            print(f"Average Order Value: ${profile['avg_order_value']:.2f}")
        
        return profiles
    
    def visualize_clusters(self):
        """Create visualizations of the clusters"""
        features = ['annual_spending', 'frequency', 'recency_days', 'avg_order_value']
        
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Scatter plots for different feature combinations
        scatter_pairs = [
            ('annual_spending', 'frequency'),
            ('annual_spending', 'avg_order_value'),
            ('frequency', 'recency_days'),
            ('avg_order_value', 'recency_days')
        ]
        
        for i, (x_feature, y_feature) in enumerate(scatter_pairs):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            for cluster_id in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cluster_id]
                ax.scatter(cluster_data[x_feature], cluster_data[y_feature], 
                          label=f'Cluster {cluster_id}', alpha=0.6, s=50)
            
            ax.set_xlabel(x_feature.replace('_', ' ').title())
            ax.set_ylabel(y_feature.replace('_', ' ').title())
            ax.set_title(f'{x_feature.replace("_", " ").title()} vs {y_feature.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Cluster size distribution
        ax = axes[1, 2]
        cluster_counts = self.data['cluster'].value_counts().sort_index()
        ax.bar(cluster_counts.index, cluster_counts.values, alpha=0.7)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Cluster Size Distribution')
        ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[0, 2].remove()
        
        plt.tight_layout()
        plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create heatmap of cluster characteristics
        features_for_heatmap = ['annual_spending', 'frequency', 'recency_days', 'avg_order_value']
        cluster_means = self.data.groupby('cluster')[features_for_heatmap].mean()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_means.T, annot=True, fmt='.1f', cmap='viridis', 
                    cbar_kws={'label': 'Average Value'})
        plt.title('Cluster Characteristics Heatmap')
        plt.xlabel('Cluster')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_customer_segment(self, customer_data):
        """Predict cluster for new customer data"""
        if self.kmeans is None:
            raise ValueError("Model not trained. Run perform_clustering() first.")
        
        # Scale the new data
        scaled_data = self.scaler.transform([customer_data])
        cluster = self.kmeans.predict(scaled_data)[0]
        
        return cluster
    
    def save_results(self, filename='customer_segments.csv'):
        """Save clustered data to CSV"""
        self.data.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    """Main function to demonstrate the clustering algorithm"""
    print("=== Customer Segmentation using K-Means Clustering ===\n")
    
    # Initialize the segmentation model
    segmenter = CustomerSegmentation()
    
    # Generate sample data (or load from file)
    print("1. Generating sample customer data...")
    data = segmenter.generate_sample_data(n_customers=1000)
    print(f"Data shape: {data.shape}")
    print("\nSample data:")
    print(data.head())
    
    # Preprocess the data
    print("\n2. Preprocessing data...")
    segmenter.preprocess_data()
    
    # Find optimal number of clusters
    print("\n3. Finding optimal number of clusters...")
    optimal_k = segmenter.find_optimal_clusters(max_k=8)
    
    # Perform clustering
    print(f"\n4. Performing K-means clustering with {optimal_k} clusters...")
    segmenter.perform_clustering(n_clusters=optimal_k)
    
    # Analyze clusters
    print("\n5. Analyzing clusters...")
    profiles = segmenter.analyze_clusters()
    
    # Visualize results
    print("\n6. Creating visualizations...")
    segmenter.visualize_clusters()
    
    # Save results
    print("\n7. Saving results...")
    segmenter.save_results()
    
    # Example: Predict cluster for a new customer
    print("\n8. Example: Predicting cluster for new customer...")
    new_customer = [2500, 30, 20, 100]  # [annual_spending, frequency, recency_days, avg_order_value]
    predicted_cluster = segmenter.get_customer_segment(new_customer)
    print(f"New customer with data {new_customer} belongs to Cluster {predicted_cluster}")
    
    print("\n=== Customer Segmentation Complete ===")

if __name__ == "__main__":
    main()