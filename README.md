## Project Explanation

**Overview:**

This project performs unsupervised clustering on aerial images of cacti. The objective is to group similar cactus species without relying on labels by leveraging deep learning for feature extraction and a robust ensemble clustering scheme. The pipeline consists of several key steps: feature extraction using a pretrained deep network, dimensionality reduction with PCA, ensemble clustering, evaluation of clustering quality, and comprehensive visualization of the results.

### How It Works

#### Feature Extraction Using Pretrained ResNet50

**What It Does:**  
Uses a pretrained ResNet50 model from PyTorch to extract high-level features from each cactus image.

**How It Works:**  
- **Preprocessing:**  
  Each image is resized to 224×224 and normalized using mean and standard deviation values that were used during the training of ResNet50.
- **Truncating the Network:**  
  The final classification layer of ResNet50 is removed so that the network outputs a feature vector instead of class predictions. This vector captures the essential visual characteristics of the image.

**Why It’s Important:**  
The pretrained network has learned useful representations from a large dataset (ImageNet), so its features can generalize well to new data, even without labels.

#### Dimensionality Reduction with PCA

**What It Does:**  
Reduces the dimensionality of the extracted feature vectors to 50 principal components.

**How It Works:**  
- **Variance Preservation:**  
  PCA transforms the high-dimensional feature space into a smaller space while retaining most of the variance (information) in the data.
- **Noise Reduction:**  
  It removes redundant or noisy information, which simplifies clustering.

**Why It’s Important:**  
Lower dimensions lead to faster and more efficient clustering while mitigating the curse of dimensionality.

#### Ensemble Clustering

**What It Does:**  
Enhances the robustness of clustering by aggregating multiple KMeans runs with different random initializations.

**How It Works:**  
- **Multiple Runs:**  
  The algorithm runs KMeans several times (e.g., 10 runs) on the PCA-reduced data.
- **Consensus Matrix:**  
  For every pair of images, a consensus score is computed to indicate how consistently they are clustered together across the ensemble.
- **Final Clustering:**  
  A final KMeans clustering is performed on the consensus matrix to produce the robust final cluster assignments.

**Why It’s Important:**  
Ensemble methods reduce the impact of any single run’s random initialization, resulting in more stable and reliable clustering results.

#### Cluster Evaluation

**Key Metrics Computed:**

- **Silhouette Score:**  
  Measures how similar each image is to its own cluster compared to other clusters. Ranges from -1 to 1 (higher is better).

- **Davies-Bouldin Score:**  
  Quantifies the average similarity between clusters (lower values indicate distinct clusters).

- **Calinski-Harabasz Score:**  
  Also known as the Variance Ratio Criterion; higher values suggest better separation.

- **Explained Variance:**  
  Indicates the percentage of variance retained after PCA.

- **Cluster Stability:**  
  Analyzed using K-Fold cross-validation to measure the consistency of clustering across different data splits.

- **Feature Importance:**  
  A Random Forest classifier is used to determine the importance of each PCA component in defining the clusters.

**Why It’s Important:**  
These metrics provide quantitative assessments of how well the clusters are formed and help in fine-tuning the clustering approach.

#### Visualization

**Components of Visualization:**

- **t-SNE Plot:**  
  A dimensionality reduction technique that projects features into 2D space. This plot gives an intuitive visual representation of how well the clusters are separated.

- **Cluster Distribution Plots:**  
  Bar charts showing the number of images in each cluster.

- **Stability and Quality Metrics:**  
  Visual bar charts that depict the overall quality and stability of the clustering.

- **Feature Importance Plot:**  
  Displays which PCA components contribute most to the clustering.

**Why It’s Important:**  
Visualizations allow for quick assessment of clustering results, making it easier to interpret and validate the unsupervised learning outcomes.

---

## Requirements

This project requires Python 3.6 or above. The necessary packages are listed in the [requirements.txt](requirements.txt) file.

Install the required packages using:

3. **Output Files:**  
   The script will generate several visualization files:
   - `tsne_visualization.png`: A 2D t-SNE visualization of the clusters.
   - `feature_importance.png`: A plot showing the importance of PCA components.
   - `comprehensive_analysis.png`: A consolidated overview of cluster quality, distribution, and stability.
   - Additional console outputs provide detailed clustering metrics and evaluations.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

Amin Amou

---

## Acknowledgements

The original training and test datasets of aerial cactus images are sourced from Kaggle. These datasets were created and published by Irving Vasquez as part of the Aerial Cactus Identification competition. The datasets can be found here: [https://www.kaggle.com/datasets/irvingvasquez/cactus-aerial-photos](https://www.kaggle.com/datasets/irvingvasquez/cactus-aerial-photos).

Thanks to the open-source communities behind PyTorch, scikit-learn, and other libraries for their invaluable contributions.
