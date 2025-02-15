import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import KFold

class CactusClusterer:
    def __init__(self, data_path, n_clusters=3):
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.n_ensemble = 10  # Number of clustering runs for ensemble
        
    def load_and_preprocess_images(self):
        features = []
        image_paths = []
        
        for img_name in os.listdir(self.data_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.data_path, img_name)
                image_paths.append(img_path)
                
                # Load and transform image
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                # Extract features
                with torch.no_grad():
                    feature = self.model(img_tensor)
                    feature = feature.squeeze().cpu().numpy()
                    features.append(feature)
        
        return np.array(features), image_paths
    
    def evaluate_clustering(self, features_pca, clusters):
        # Calculate various cluster quality metrics
        metrics = {
            'silhouette': silhouette_score(features_pca, clusters),
            'davies_bouldin': davies_bouldin_score(features_pca, clusters),
            'calinski_harabasz': calinski_harabasz_score(features_pca, clusters)
        }
        
        # Print detailed evaluation
        print("\nClustering Evaluation Metrics:")
        print("-" * 30)
        print(f"Silhouette Score: {metrics['silhouette']:.3f}")
        print(f"Davies-Bouldin Score: {metrics['davies_bouldin']:.3f}")
        print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.3f}")
        
        return metrics
    
    def visualize_tsne(self, features, clusters, metrics, filename='tsne_visualization.png'):
        """Add t-SNE visualization"""
        print("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                            c=clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Clusters')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.savefig(filename)
        plt.close()
        
    def perform_ensemble_clustering(self, features_pca):
        """Implement ensemble clustering"""
        print("Performing ensemble clustering...")
        ensemble_predictions = np.zeros((len(features_pca), self.n_ensemble))
        
        for i in range(self.n_ensemble):
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=i)
            ensemble_predictions[:, i] = kmeans.fit_predict(features_pca)
            
        # Create consensus matrix
        consensus_matrix = np.zeros((len(features_pca), len(features_pca)))
        for i in range(len(features_pca)):
            for j in range(len(features_pca)):
                consensus_matrix[i,j] = np.mean(ensemble_predictions[i,:] == ensemble_predictions[j,:])
                
        # Final clustering on consensus matrix
        final_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        consensus_clusters = final_kmeans.fit_predict(consensus_matrix)
        
        return consensus_clusters, consensus_matrix
    
    def analyze_cluster_stability(self, features_pca):
        """Add cluster stability analysis"""
        print("Analyzing cluster stability...")
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        stability_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(features_pca)):
            # Train clustering
            train_features = features_pca[train_idx]
            val_features = features_pca[val_idx]
            
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            kmeans.fit(train_features)
            
            # Get cluster assignments
            train_clusters = kmeans.predict(train_features)
            val_clusters = kmeans.predict(val_features)
            
            # Calculate stability metrics
            train_silhouette = silhouette_score(train_features, train_clusters)
            val_silhouette = silhouette_score(val_features, val_clusters)
            stability = 1 - abs(train_silhouette - val_silhouette)
            stability_scores.append(stability)
            
            print(f"Fold {fold+1} Stability: {stability:.3f}")
            
        return np.mean(stability_scores), np.std(stability_scores)
    
    def analyze_feature_importance(self, features_pca, clusters):
        """Add feature importance analysis"""
        print("Analyzing feature importance...")
        
        # Train a Random Forest classifier to identify important features
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features_pca, clusters)
        
        # Get feature importance scores
        importance_scores = rf.feature_importances_
        
        # Visualize feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance_scores)), importance_scores)
        plt.title('Feature Importance Analysis')
        plt.xlabel('PCA Component')
        plt.ylabel('Importance Score')
        plt.savefig('feature_importance.png')
        plt.close()
        
        return importance_scores
    
    def cluster_images(self):
        # Extract features
        print("Extracting features...")
        features, image_paths = self.load_and_preprocess_images()
        features = features.reshape(features.shape[0], -1)
        
        # Apply PCA
        print("Applying PCA...")
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features)
        
        # Calculate explained variance
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"Explained variance by PCA: {explained_variance:.2%}")
        
        # Perform ensemble clustering
        consensus_clusters, consensus_matrix = self.perform_ensemble_clustering(features_pca)
        
        # Analyze cluster stability
        stability_mean, stability_std = self.analyze_cluster_stability(features_pca)
        
        # Analyze feature importance
        importance_scores = self.analyze_feature_importance(features_pca, consensus_clusters)
        
        # Visualize with t-SNE
        self.visualize_tsne(features_pca, consensus_clusters, None)
        
        # Calculate other metrics
        metrics = {
            'silhouette': silhouette_score(features_pca, consensus_clusters),
            'davies_bouldin': davies_bouldin_score(features_pca, consensus_clusters),
            'calinski_harabasz': calinski_harabasz_score(features_pca, consensus_clusters),
            'stability_mean': stability_mean,
            'stability_std': stability_std,
            'explained_variance': explained_variance,
            'cluster_sizes': Counter(consensus_clusters),
            'feature_importance': importance_scores
        }
        
        return consensus_clusters, image_paths, metrics
    
    def visualize_all_results(self, clusters, image_paths, metrics):
        """Create comprehensive visualization of all results"""
        plt.figure(figsize=(20, 15))
        
        # 1. Cluster Distribution
        plt.subplot(3, 2, 1)
        cluster_sizes = metrics['cluster_sizes']
        plt.bar(range(self.n_clusters), [cluster_sizes[i] for i in range(self.n_clusters)])
        plt.title('Cluster Distribution')
        
        # 2. Feature Importance
        plt.subplot(3, 2, 2)
        plt.bar(range(len(metrics['feature_importance'])), metrics['feature_importance'])
        plt.title('Feature Importance')
        
        # 3. Stability Analysis
        plt.subplot(3, 2, 3)
        plt.bar(['Mean Stability'], [metrics['stability_mean']])
        plt.errorbar(['Mean Stability'], [metrics['stability_mean']], 
                    yerr=[metrics['stability_std']], fmt='none', color='black')
        plt.title('Cluster Stability')
        
        # 4. Quality Metrics
        plt.subplot(3, 2, 4)
        quality_metrics = {
            'Silhouette': metrics['silhouette'],
            'Davies-Bouldin': metrics['davies_bouldin'],
            'Calinski-Harabasz': metrics['calinski_harabasz']/1000  # Scaled for visualization
        }
        plt.bar(quality_metrics.keys(), quality_metrics.values())
        plt.title('Quality Metrics')
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png')
        plt.close()

def main():
    # Initialize and run clustering with k=3
    clusterer = CactusClusterer('train', n_clusters=3)
    clusters, image_paths, metrics = clusterer.cluster_images()
    
    # Visualize all results
    clusterer.visualize_all_results(clusters, image_paths, metrics)
    
    # Print comprehensive analysis
    print("\nComprehensive Clustering Analysis:")
    print("=" * 50)
    print(f"\n1. Cluster Stability: {metrics['stability_mean']:.3f} ± {metrics['stability_std']:.3f}")
    print(f"2. Silhouette Score: {metrics['silhouette']:.3f}")
    print(f"3. Explained Variance: {metrics['explained_variance']:.2%}")
    print("\n4. Cluster Sizes:")
    for cluster_id, size in metrics['cluster_sizes'].items():
        print(f"   Cluster {cluster_id}: {size} images")
    
    print("\n5. Top 5 Most Important Features:")
    top_features = np.argsort(metrics['feature_importance'])[-5:]
    for idx, feature in enumerate(reversed(top_features)):
        print(f"   {idx+1}. Component {feature}: {metrics['feature_importance'][feature]:.3f}")

if __name__ == "__main__":
    main() 