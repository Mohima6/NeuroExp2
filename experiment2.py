import numpy as np
import pandas as pd
from scipy.linalg import cholesky, logm, sqrtm, inv
from scipy.stats import wishart
from sklearn.metrics import accuracy_score
import os
import time
import warnings
warnings.filterwarnings('ignore')
n_subjects = 100
n_sessions = 4
n_regions = 65
n_networks = 7
network_sizes = [10, 8, 12, 9, 11, 7, 8]
region_to_network = np.repeat(np.arange(n_networks), network_sizes)
np.random.seed(123)
data_files_npy = ['fingerprint_correlations.npy',
                  'fingerprint_subject_ids.npy',
                  'fingerprint_session_ids.npy']
data_files_csv = ['fingerprint_data.csv']
start_time = time.time()
if all(os.path.exists(f) for f in data_files_npy):
    print("Loading NPY files")
    all_corr = np.load('fingerprint_correlations.npy')
    subject_ids = np.load('fingerprint_subject_ids.npy')
    session_ids = np.load('fingerprint_session_ids.npy')
elif os.path.exists('fingerprint_data.csv'):
    print("Loading from CSV")
    df = pd.read_csv('fingerprint_data.csv')
    subject_ids = df['subject_id'].values
    session_ids = df['session_id'].values
    n_samples = len(df)
    i_upper = np.triu_indices(n_regions, k=1)
    all_corr = np.zeros((n_samples, n_regions, n_regions))
    edge_cols = [c for c in df.columns if c not in ['subject_id', 'session_id']]
    X_flat = df[edge_cols].values
    for subj in range(n_samples):
        if subj % 50 == 0:
            print(f"  Reconstructing matrix {subj}/{n_samples}")
        mat = np.zeros((n_regions, n_regions))
        mat[i_upper] = X_flat[subj]
        mat = mat + mat.T
        np.fill_diagonal(mat, 1.0)
        all_corr[subj] = mat
    np.save('fingerprint_correlations.npy', all_corr)
    np.save('fingerprint_subject_ids.npy', subject_ids)
    np.save('fingerprint_session_ids.npy', session_ids)
    print("Converted CSV to NPY.")
else:
    print("synthetic data")
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
    print("subject‑specific true FCs")
    subject_true = []
    for s in range(n_subjects):
        if s % 20 == 0:
            print(f"    Subject {s}/{n_subjects}")
        subj_corr = base_corr.copy()
        perturbation = 0.1 * np.random.randn(n_regions, n_regions)
        perturbation = (perturbation + perturbation.T) / 2
        np.fill_diagonal(perturbation, 0)
        subj_corr += perturbation
        subj_corr = np.clip(subj_corr, -1, 1)
        np.fill_diagonal(subj_corr, 1.0)
        eigvals = np.linalg.eigvalsh(subj_corr)
        if np.min(eigvals) < 1e-6:
            subj_corr += (1e-6 - np.min(eigvals)) * np.eye(n_regions)
        subject_true.append(subj_corr)
    subject_true = np.array(subject_true)
    print("session‑specific FCs")
    all_corr = np.zeros((n_subjects, n_sessions, n_regions, n_regions))
    for s in range(n_subjects):
        if s % 10 == 0:
            print(f"    Subject {s}/{n_subjects}")
        for sess in range(n_sessions):
            sess_corr = subject_true[s].copy()
            dof = n_regions + 10
            scale = sess_corr * dof
            eigvals_scale = np.linalg.eigvalsh(scale)
            if np.min(eigvals_scale) < 1e-6:
                scale += (1e-6 - np.min(eigvals_scale)) * np.eye(n_regions)
            cov = wishart.rvs(df=dof, scale=scale / dof, random_state=s*10 + sess)
            d = np.sqrt(np.diag(cov))
            noisy_corr = cov / np.outer(d, d)
            all_corr[s, sess] = noisy_corr
    all_corr = all_corr.reshape(-1, n_regions, n_regions)
    subject_ids = np.repeat(np.arange(n_subjects), n_sessions)
    session_ids = np.tile(np.arange(n_sessions), n_subjects)
    np.save('fingerprint_correlations.npy', all_corr)
    np.save('fingerprint_subject_ids.npy', subject_ids)
    np.save('fingerprint_session_ids.npy', session_ids)
    print("Saving data to CSV")
    i_upper = np.triu_indices(n_regions, k=1)
    X_flat = np.array([corr[i_upper] for corr in all_corr])
    col_names = [f"edge_{i}_{j}" for i, j in zip(i_upper[0], i_upper[1])]
    df = pd.DataFrame(X_flat, columns=col_names)
    df.insert(0, 'session_id', session_ids)
    df.insert(0, 'subject_id', subject_ids)
    df.to_csv('fingerprint_data.csv', index=False)
    print("Data generated and saved")
load_time = time.time() - start_time
print(f"Data ready. Loading took {load_time:.1f} seconds.")
def vec_pearson_dist(A, B):
    i_upper = np.triu_indices_from(A, k=1)
    vA = A[i_upper]
    vB = B[i_upper]
    corr = np.corrcoef(vA, vB)[0, 1]
    return 1 - corr
def euclidean_cholesky_dist(A, B):
    L_A = cholesky(A, lower=True)
    L_B = cholesky(B, lower=True)
    i_lower = np.tril_indices_from(L_A)
    return np.linalg.norm(L_A[i_lower] - L_B[i_lower])
def log_euclidean_dist(A, B):
    return np.linalg.norm(logm(A) - logm(B), ord='fro')
def airm_dist(A, B):
    A_sqrt = sqrtm(A)
    A_sqrt_inv = inv(A_sqrt)
    M = A_sqrt_inv @ B @ A_sqrt_inv
    return np.linalg.norm(logm(M), ord='fro')
metrics = {
    'Baseline (Pearson)': vec_pearson_dist,
    'ECM': euclidean_cholesky_dist,
    'LEC': log_euclidean_dist,
    'AIRM': airm_dist
}
session_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
results = {name: [] for name in metrics}
print("\nStarting fingerprinting")
total_pairs = len(session_pairs)
pair_count = 0
for s1, s2 in session_pairs:
    pair_count += 1
    print(f"\nProcessing pair {pair_count}/{total_pairs}: ({s1},{s2})")
    idx_g1 = np.where(session_ids == s1)[0]
    idx_p1 = np.where(session_ids == s2)[0]
    gallery1 = all_corr[idx_g1]
    probe1 = all_corr[idx_p1]
    subj_g1 = subject_ids[idx_g1]
    subj_p1 = subject_ids[idx_p1]
    idx_g2 = np.where(session_ids == s2)[0]
    idx_p2 = np.where(session_ids == s1)[0]
    gallery2 = all_corr[idx_g2]
    probe2 = all_corr[idx_p2]
    subj_g2 = subject_ids[idx_g2]
    subj_p2 = subject_ids[idx_p2]
    for name, dist_func in metrics.items():
        print(f"  Metric: {name}")
        # Order 1
        pred1 = []
        n_probe = len(probe1)
        for p in range(n_probe):
            if p % 20 == 0:
                print(f"    Probe {p}/{n_probe} (order 1)")
            dists = [dist_func(probe1[p], gallery1[g]) for g in range(len(gallery1))]
            pred1.append(subj_g1[np.argmin(dists)])
        acc1 = accuracy_score(subj_p1, pred1)
        # Order 2
        pred2 = []
        n_probe2 = len(probe2)
        for p in range(n_probe2):
            if p % 20 == 0:
                print(f"    Probe {p}/{n_probe2} (order 2)")
            dists = [dist_func(probe2[p], gallery2[g]) for g in range(len(gallery2))]
            pred2.append(subj_g2[np.argmin(dists)])
        acc2 = accuracy_score(subj_p2, pred2)
        acc_avg = (acc1 + acc2) / 2
        results[name].append(acc_avg)
output_lines = []
output_lines.append("="*75)
output_lines.append(" Accuracy (averaged over session pairs)")
output_lines.append("="*75)
header = f"{'Metric':<25} {'Pair(0,1)':<10} {'Pair(0,2)':<10} {'Pair(0,3)':<10} " \
        f"{'Pair(1,2)':<10} {'Pair(1,3)':<10} {'Pair(2,3)':<10} {'Mean':<10}"
output_lines.append(header)
output_lines.append("-"*105)
for name in metrics:
    row = f"{name:<25}"
    for v in results[name]:
        row += f" {v*100:6.2f}%"
    mean_acc = np.mean(results[name]) * 100
    row += f" {mean_acc:6.2f}%"
    output_lines.append(row)
output_lines.append("="*75)
print("\n" + "\n".join(output_lines))
with open('fingerprinting_results.txt', 'w') as f:
    for line in output_lines:
        f.write(line + '\n')
print("\nResults saved to 'fingerprinting_results.txt'.")
print("\nData files:")
print("  - fingerprint_correlations.npy")
print("  - fingerprint_data.csv")
print("  - fingerprinting_results.txt")
