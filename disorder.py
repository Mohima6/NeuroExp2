import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import cholesky
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import networkx as nx
import os
import warnings
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
print("Loading data")
df_corr = pd.read_csv('synthetic_correlations.csv')
df_pmat = pd.read_csv('synthetic_pmat.csv')
X_flat = df_corr.values.astype(np.float64)                
pmat_cr = df_pmat['PMAT_A_CR'].values.astype(np.float64)
n_subjects, n_edges = X_flat.shape
n_regions = 65
print("correlation matrices")
i_upper = np.triu_indices(n_regions, k=1)
all_corr = np.zeros((n_subjects, n_regions, n_regions))
for subj in range(n_subjects):
    if subj % 200 == 0:
        print(f"  Subject {subj}/{n_subjects}")
    mat = np.zeros((n_regions, n_regions))
    mat[i_upper] = X_flat[subj, :]
    mat = mat + mat.T
    np.fill_diagonal(mat, 1.0)
    all_corr[subj] = mat
print(f"Loaded {n_subjects} correlation matrices.")
median = np.median(pmat_cr)
labels = (pmat_cr < median).astype(int)
print(f"Class distribution: 0 (high) = {sum(labels==0)}, 1 (low) = {sum(labels==1)}")
network_sizes = [10, 8, 12, 9, 11, 7, 8]
region_to_network = np.repeat(np.arange(7), network_sizes)
def lec_vectorize(mat):
    L = cholesky(mat, lower=True)
    i_lower = np.tril_indices_from(L)
    return L[i_lower]
print("Applying LEC transformation")
X = np.array([lec_vectorize(m) for m in all_corr])
y = labels
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
models = {
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gaussian Process': GaussianProcessClassifier(kernel=RBF(), random_state=42)
}
param_grids = {
    'SVM': {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
    'Gaussian Process': {}  # use default kernel
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
test_probs = {}
test_preds = {}
test_accuracies = {}
print("\nTraining models")
for name in models:
    print(f"\n{name}:")
    if param_grids[name]:
        grid = GridSearchCV(models[name], param_grids[name], cv=cv, scoring='roc_auc')
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
    else:
        best_models[name] = models[name].fit(X_train, y_train)
    test_probs[name] = best_models[name].predict_proba(X_test)[:, 1]
    test_preds[name] = best_models[name].predict(X_test)
    test_accuracies[name] = accuracy_score(y_test, test_preds[name])
    print(f"  Test accuracy: {test_accuracies[name]:.3f}")
print("\nfigures")
# 1. Group-avg. FC heatmaps 
mean_high = np.mean(all_corr[labels==0], axis=0)
mean_low = np.mean(all_corr[labels==1], axis=0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(mean_high, ax=axes[0], cmap='RdBu_r', center=0,
            xticklabels=False, yticklabels=False, cbar=True)
axes[0].set_title('High PMAT group average FC')
sns.heatmap(mean_low, ax=axes[1], cmap='RdBu_r', center=0,
            xticklabels=False, yticklabels=False, cbar=True)
axes[1].set_title('Low PMAT group average FC')
plt.tight_layout()
plt.savefig('figures/figure1_group_averages.png', dpi=150)
plt.close()
# 2. Diff. matrix with network boundaries 
diff_mat = np.abs(mean_high - mean_low)
network_order = np.argsort(region_to_network)
diff_sorted = diff_mat[np.ix_(network_order, network_order)]
plt.figure(figsize=(8, 7))
sns.heatmap(diff_sorted, cmap='Reds', cbar_kws={'label': 'Absolute difference'},
            xticklabels=False, yticklabels=False, square=True)
boundaries = [0]
current_net = region_to_network[network_order[0]]
for i, net in enumerate(region_to_network[network_order]):
    if net != current_net:
        boundaries.append(i)
        current_net = net
boundaries.append(n_regions)
for b in boundaries:
    plt.axhline(b, color='blue', linewidth=1)
    plt.axvline(b, color='blue', linewidth=1)
plt.title('Group difference (High vs Low PMAT) with network boundaries')
plt.tight_layout()
plt.savefig('figures/figure2_difference_matrix.png', dpi=150)
plt.close()
# 3. Brain network graph (top 2% edges of high PMAT group) 
G = nx.Graph()
threshold = np.percentile(np.abs(mean_high[np.triu_indices_from(mean_high, k=1)]), 98)
for i in range(n_regions):
    G.add_node(i, network=region_to_network[i])
pos = nx.spring_layout(G, seed=42, iterations=50)
for i in range(n_regions):
    for j in range(i+1, n_regions):
        if np.abs(mean_high[i, j]) >= threshold:
            G.add_edge(i, j, weight=mean_high[i, j])
plt.figure(figsize=(10, 8))
colors = [region_to_network[n] for n in G.nodes]
nx.draw(G, pos, node_color=colors, cmap='tab10', with_labels=False,
        node_size=50, edge_color='gray', alpha=0.7)
plt.title('Top 2% edges of high PMAT group FC')
plt.savefig('figures/figure3_brain_graph.png', dpi=150)
plt.close()
# 4. PCA projection 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.colorbar(scatter, ticks=[0, 1], label='Group (0=High, 1=Low)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of LEC-transformed FCs')
plt.savefig('figures/figure4_pca.png', dpi=150)
plt.close()
# 5. t-SNE embedding 
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.colorbar(scatter, ticks=[0, 1], label='Group')
plt.title('t-SNE of LEC-transformed FCs')
plt.savefig('figures/figure5_tsne.png', dpi=150)
plt.close()
# 6. Enhanced distance clustermap with group colors 
D = pairwise_distances(X_scaled, metric='euclidean')
group_colors = ['skyblue' if g == 0 else 'salmon' for g in y]
g = sns.clustermap(D, cmap='viridis', method='average',
                   figsize=(10, 9),
                   cbar_kws={'label': 'Euclidean distance (LEC space)'},
                   row_colors=group_colors, col_colors=group_colors,
                   xticklabels=False, yticklabels=False)
legend_elements = [Patch(facecolor='skyblue', label='High PMAT (healthy)'),
                   Patch(facecolor='salmon', label='Low PMAT (disorder)')]
g.ax_heatmap.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
g.ax_heatmap.set_title('Pairwise distances with hierarchical clustering')
plt.savefig('figures/figure6_distance_clustermap_annotated.png', dpi=150)
plt.close()
# 7. Confusion matrices 
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (name, preds) in enumerate(test_preds.items()):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['High', 'Low'])
    disp.plot(ax=axes[i], colorbar=False)
    axes[i].set_title(f'{name}')
plt.tight_layout()
plt.savefig('figures/figure7_confusion_matrices.png', dpi=150)
plt.close()
# 8. ROC curves 
plt.figure(figsize=(8, 6))
for name in test_probs:
    RocCurveDisplay.from_predictions(y_test, test_probs[name], name=name, ax=plt.gca())
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.title('ROC Curves') 
plt.legend()
plt.savefig('figures/figure8_roc_curves.png', dpi=150)
plt.close()
# 9. Feature importance
rf = best_models['Random Forest']
importances = rf.feature_importances_
top_idx = np.argsort(importances)[-20:]
plt.figure(figsize=(8, 5))
plt.barh(range(20), importances[top_idx])
plt.yticks(range(20), [f'Feat {i}' for i in top_idx])
plt.xlabel('Importance')
plt.title('Top 20 features (Random Forest)')
plt.tight_layout()
plt.savefig('figures/figure9_feature_importance.png', dpi=150)
plt.close()
# 10. Model accuracy comparison 
accuracies = [test_accuracies[name] for name in models]
plt.figure(figsize=(6, 5))
bars = plt.bar(models.keys(), accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(0, 1)
plt.ylabel('Test Accuracy')
plt.title('Model Comparison')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center')
plt.savefig('figures/figure10_model_comparison.png', dpi=150)
plt.close()
# 11. Geometry comparison
labels_all = y
within_dists = []
between_dists = []
for i in range(len(X_scaled)):
    for j in range(i+1, len(X_scaled)):
        d = np.linalg.norm(X_scaled[i] - X_scaled[j])
        if labels_all[i] == labels_all[j]:
            within_dists.append(d)
        else:
            between_dists.append(d)
plt.figure(figsize=(6, 5))
plt.boxplot([within_dists, between_dists], labels=['Within-class', 'Between-class'])
plt.ylabel('Euclidean distance (LEC space)')
plt.title('Within- vs between-class distances')
plt.savefig('figures/figure11_distance_boxplot.png', dpi=150)
plt.close()
# 12. 3D brain visualization (top 10 discriminative connections)
try:
    from nilearn import plotting
    coords = np.random.randn(n_regions, 3) * 10
    diff = np.abs(mean_high - mean_low)
    i_upper = np.triu_indices_from(diff, k=1)
    edge_weights = diff[i_upper]
    top_edges = np.argsort(edge_weights)[-10:]
    adj = np.zeros((n_regions, n_regions))
    for idx in top_edges:
        i, j = i_upper[0][idx], i_upper[1][idx]
        adj[i, j] = edge_weights[idx]
        adj[j, i] = edge_weights[idx]
    plotting.plot_connectome(adj, coords, edge_threshold='95%',
                             title='Top 10 discriminative connections',
                             display_mode='ortho')
    plt.savefig('figures/figure12_3d_brain.png', dpi=150)
    plt.close()
except ImportError:
    print("nilearn not installed.")
# 13. Individual fingerprint matrices 
n_examples = 4  
healthy_idx = np.where(y == 0)[0][:n_examples]
disorder_idx = np.where(y == 1)[0][:n_examples]
fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 8))
fig.suptitle('Individual FC matrices: Healthy (top) vs Disorder (bottom)', fontsize=16)
for i, idx in enumerate(healthy_idx):
    sns.heatmap(all_corr[idx], ax=axes[0, i], cmap='RdBu_r', center=0,
                xticklabels=False, yticklabels=False, cbar=False)
    axes[0, i].set_title(f'Healthy #{i + 1}')
for i, idx in enumerate(disorder_idx):
    sns.heatmap(all_corr[idx], ax=axes[1, i], cmap='RdBu_r', center=0,
                xticklabels=False, yticklabels=False, cbar=False)
    axes[1, i].set_title(f'Disorder #{i + 1}')
plt.tight_layout()
plt.savefig('figures/figure13_individual_fingerprints.png', dpi=150)
plt.close()
print("\nAll figures saved in 'figures/' directory.")
print("\n" + "="*50)
print("Test Set Accuracies")
print("="*50)
for name in models:
    print(f"{name:20}: {test_accuracies[name]:.4f}")
print("="*50)
# Save to a text file
with open('classification_results.txt', 'w') as f:
    f.write("Test Set Accuracies\n")
    f.write("="*30 + "\n")
    for name in models:
        f.write(f"{name:20}: {test_accuracies[name]:.4f}\n")
print("\nResults also saved to 'classification_results.txt'.")
