import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import cholesky, logm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score, RocCurveDisplay,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import pairwise_distances
import networkx as nx
import os
from matplotlib.patches import Patch
from scipy.stats import wishart
import warnings
warnings.filterwarnings('ignore')
n_subjects_per_group = 200
n_groups = 5
group_names = ['Healthy', 'Alzheimer', 'Autism', 'Parkinson', 'Schizophrenia']
n_subjects = n_subjects_per_group * n_groups
n_regions = 65
n_networks = 7
network_sizes = [10, 8, 12, 9, 11, 7, 8]
region_to_network = np.repeat(np.arange(n_networks), network_sizes)
np.random.seed(123)
print("correlation matrix")
base_corr = np.eye(n_regions)
for net in range(n_networks):
    mask = region_to_network == net
    block = 0.5 + 0.1 * np.random.randn(np.sum(mask), np.sum(mask))
    block = (block + block.T) / 2
    np.fill_diagonal(block, 1.0)
    base_corr[np.ix_(mask, mask)] = block
for i in range(n_regions):
    for j in range(i+1, n_regions):
        if region_to_network[i] != region_to_network[j]:
            base_corr[i, j] = 0.1 + 0.05 * np.random.randn()
            base_corr[j, i] = base_corr[i, j]
eigvals = np.linalg.eigvalsh(base_corr)
if np.min(eigvals) < 1e-6:
    base_corr += (1e-6 - np.min(eigvals)) * np.eye(n_regions)
alterations = {
    'Healthy': {0:1.0, 1:1.0, 2:1.0, 3:1.0},  
    'Alzheimer': {0:0.7, 1:1.0, 2:1.0, 3:1.0},  
    'Autism': {0:1.0, 1:0.7, 2:1.0, 3:1.0},     
    'Parkinson': {0:1.0, 1:1.0, 2:0.7, 3:1.0},   
    'Schizophrenia': {0:1.0, 1:1.0, 2:1.0, 3:0.7} 
}
all_corr = []
labels = []
subject_group = []
for g, group in enumerate(group_names):
    print(f"Generating {group}")
    for s in range(n_subjects_per_group):
        subj_corr = base_corr.copy()
        for net, factor in alterations[group].items():
            mask = region_to_network == net
            subj_corr[np.ix_(mask, mask)] *= factor
        perturbation = 0.1 * np.random.randn(n_regions, n_regions)
        perturbation = (perturbation + perturbation.T) / 2
        np.fill_diagonal(perturbation, 0)
        subj_corr += perturbation
        subj_corr = np.clip(subj_corr, -1, 1)
        np.fill_diagonal(subj_corr, 1.0)
        eigvals = np.linalg.eigvalsh(subj_corr)
        if np.min(eigvals) < 1e-6:
            subj_corr += (1e-6 - np.min(eigvals)) * np.eye(n_regions)
        dof = n_regions + 10
        scale = subj_corr * dof
        eigvals_scale = np.linalg.eigvalsh(scale)
        if np.min(eigvals_scale) < 1e-6:
            scale += (1e-6 - np.min(eigvals_scale)) * np.eye(n_regions)
        cov = wishart.rvs(df=dof, scale=scale/dof, random_state=g*1000 + s)
        d = np.sqrt(np.diag(cov))
        noisy_corr = cov / np.outer(d, d)
        all_corr.append(noisy_corr)
        labels.append(g)
        subject_group.append(group)
all_corr = np.array(all_corr)
labels = np.array(labels)
print(f"Generated {len(all_corr)} subjects.")
np.save('multi_disorder_correlations.npy', all_corr)
np.save('multi_disorder_labels.npy', labels)
i_upper = np.triu_indices(n_regions, k=1)
X_flat = np.array([corr[i_upper] for corr in all_corr])
col_names = [f"edge_{i}_{j}" for i,j in zip(i_upper[0], i_upper[1])]
df = pd.DataFrame(X_flat, columns=col_names)
df.insert(0, 'group', subject_group)
df.insert(0, 'label', labels)
df.to_csv('multi_disorder_data.csv', index=False)
print("Data saved.")
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
    'SVM': OneVsRestClassifier(SVC(probability=True, random_state=42)),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gaussian Process': OneVsRestClassifier(GaussianProcessClassifier(kernel=RBF(), random_state=42))
}
param_grids = {
    'SVM': {'estimator__C': [0.1, 1, 10], 'estimator__gamma': ['scale', 0.01, 0.1]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
    'Gaussian Process': {}  
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
        grid = GridSearchCV(models[name], param_grids[name], cv=cv, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
    else:
        best_models[name] = models[name].fit(X_train, y_train)
    test_preds[name] = best_models[name].predict(X_test)
    if hasattr(best_models[name], 'predict_proba'):
        test_probs[name] = best_models[name].predict_proba(X_test)
    test_accuracies[name] = accuracy_score(y_test, test_preds[name])
    print(f"  Test accuracy: {test_accuracies[name]:.3f}")
    print(classification_report(y_test, test_preds[name], target_names=group_names))
os.makedirs('figures', exist_ok=True)
print("\nfigures")
# 1. Group-avg. FC heatmaps
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for g, group in enumerate(group_names):
    mean_mat = np.mean(all_corr[labels==g], axis=0)
    sns.heatmap(mean_mat, ax=axes[g], cmap='RdBu_r', center=0,
                xticklabels=False, yticklabels=False, cbar=True)
    axes[g].set_title(f'{group} average')
axes[5].axis('off')
plt.tight_layout()
plt.savefig('figures/figure1_group_averages.png', dpi=150)
plt.close()
# 2. Difference matrices
healthy_mean = np.mean(all_corr[labels==0], axis=0)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, g in enumerate([1,2,3,4]):
    disorder_mean = np.mean(all_corr[labels==g], axis=0)
    diff = np.abs(healthy_mean - disorder_mean)
    sns.heatmap(diff, ax=axes[idx], cmap='Reds',
                xticklabels=False, yticklabels=False, cbar=True)
    axes[idx].set_title(f'{group_names[g]} vs Healthy')
plt.tight_layout()
plt.savefig('figures/figure2_difference_matrices.png', dpi=150)
plt.close()
# 3. Brain network graph
for g, group in enumerate(group_names):
    mean_mat = np.mean(all_corr[labels==g], axis=0)
    G = nx.Graph()
    threshold = np.percentile(np.abs(mean_mat[np.triu_indices_from(mean_mat, k=1)]), 98)
    for i in range(n_regions):
        G.add_node(i, network=region_to_network[i])
    pos = nx.spring_layout(G, seed=42, iterations=50)
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            if np.abs(mean_mat[i, j]) >= threshold:
                G.add_edge(i, j, weight=mean_mat[i, j])
    plt.figure(figsize=(10, 8))
    colors = [region_to_network[n] for n in G.nodes]
    nx.draw(G, pos, node_color=colors, cmap='tab10', with_labels=False,
            node_size=50, edge_color='gray', alpha=0.7)
    plt.title(f'Top 2% edges – {group}')
    plt.savefig(f'figures/figure3_graph_{group}.png', dpi=150)
    plt.close()
# 4. PCA 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', edgecolor='k', alpha=0.7)
plt.colorbar(scatter, ticks=range(5), label='Group')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of LEC-transformed FCs')
plt.savefig('figures/figure4_pca.png', dpi=150)
plt.close()
# 5. t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', edgecolor='k', alpha=0.7)
plt.colorbar(scatter, ticks=range(5), label='Group')
plt.title('t-SNE of LEC-transformed FCs')
plt.savefig('figures/figure5_tsne.png', dpi=150)
plt.close()
# 6. Distance clustermap 
D = pairwise_distances(X_scaled, metric='euclidean')
group_colors = [plt.cm.tab10(g/5) for g in y]
g = sns.clustermap(D, cmap='viridis', method='average',
                   figsize=(12, 11),
                   cbar_kws={'label': 'Euclidean distance'},
                   row_colors=group_colors, col_colors=group_colors,
                   xticklabels=False, yticklabels=False)
legend_elements = [Patch(facecolor=plt.cm.tab10(i/5), label=group_names[i]) for i in range(5)]
g.ax_heatmap.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
g.ax_heatmap.set_title('Pairwise distances with hierarchical clustering')
plt.savefig('figures/figure6_clustermap.png', dpi=150)
plt.close()
# 7. Confusion matrix 
cm = confusion_matrix(y_test, test_preds['Random Forest'])
disp = ConfusionMatrixDisplay(cm, display_labels=group_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.savefig('figures/figure7_confusion_matrix.png', dpi=150)
plt.close()
# 8. ROC curves 
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=range(5))
for name in test_probs:
    plt.figure(figsize=(8, 6))
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs[name][:, i])
        auc = roc_auc_score(y_test_bin[:, i], test_probs[name][:, i])
        plt.plot(fpr, tpr, label=f'{group_names[i]} (AUC={auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves – {name}')
    plt.legend()
    plt.savefig(f'figures/figure8_roc_{name}.png', dpi=150)
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
# 11. Within‑ vs between‑class distances
within_dists = []
between_dists = []
for i in range(len(X_scaled)):
    for j in range(i+1, len(X_scaled)):
        d = np.linalg.norm(X_scaled[i] - X_scaled[j])
        if y[i] == y[j]:
            within_dists.append(d)
        else:
            between_dists.append(d)
plt.figure(figsize=(6,5))
plt.boxplot([within_dists, between_dists], labels=['Within-class', 'Between-class'])
plt.ylabel('Euclidean distance')
plt.title('Within‑ vs between‑class distances')
plt.savefig('figures/figure11_distance_boxplot.png', dpi=150)
plt.close()
# 12. brain plots 
try:
    from nilearn import plotting
    coords = np.random.randn(n_regions, 3) * 10  
    healthy_mean = np.mean(all_corr[labels==0], axis=0)
    for g in [1,2,3,4]:
        disorder_mean = np.mean(all_corr[labels==g], axis=0)
        diff = np.abs(healthy_mean - disorder_mean)
        i_upper = np.triu_indices_from(diff, k=1)
        edge_weights = diff[i_upper]
        top_edges = np.argsort(edge_weights)[-10:]
        adj = np.zeros((n_regions, n_regions))
        for idx in top_edges:
            i, j = i_upper[0][idx], i_upper[1][idx]
            adj[i, j] = edge_weights[idx]
            adj[j, i] = edge_weights[idx]
        plotting.plot_connectome(adj, coords, edge_threshold='95%',
                                 title=f'Top 10 discriminative connections – {group_names[g]}',
                                 display_mode='ortho')
        plt.savefig(f'figures/figure12_3d_{group_names[g]}.png', dpi=150)
        plt.close()
except ImportError:
    print("nilearn not installed.")
# 13. Individual fingerprint matrices 
n_examples = 3
fig, axes = plt.subplots(n_groups, n_examples, figsize=(3*n_examples, 2*n_groups))
for g in range(n_groups):
    idxs = np.where(y == g)[0][:n_examples]
    for k, idx in enumerate(idxs):
        sns.heatmap(all_corr[idx], ax=axes[g, k], cmap='RdBu_r', center=0,
                    xticklabels=False, yticklabels=False, cbar=False)
        axes[g, k].set_title(f'{group_names[g]} #{k+1}')
plt.tight_layout()
plt.savefig('figures/figure13_individual_fingerprints.png', dpi=150)
plt.close()
print("\nfigures saved in 'figures/' folder.")
def predict_new_subject(new_corr_matrix, model, scaler, group_names):
    """
    new_corr_matrix: 65x65 correlation matrix
    model: trained classifier (e.g., best_models['Random Forest'])
    scaler: fitted StandardScaler
    """
    L = cholesky(new_corr_matrix, lower=True)
    i_lower = np.tril_indices_from(L)
    feat = L[i_lower].reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred = model.predict(feat_scaled)[0]
    proba = model.predict_proba(feat_scaled)[0]
    print(f"Predicted group: {group_names[pred]} (probability: {proba[pred]:.3f})")
    return group_names[pred], proba
test_idx = np.where(y_test == 2)[0][0]  
new_mat = all_corr[test_idx]  
print("\nPrediction demo:")
predict_new_subject(new_mat, best_models['Random Forest'], scaler, group_names)
import joblib
joblib.dump(best_models['Random Forest'], 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved.")
