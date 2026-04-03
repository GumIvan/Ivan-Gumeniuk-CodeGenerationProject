import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
 
# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('Mall_Customers.csv')
df.columns = ['CustomerID', 'Gender', 'Age', 'Income', 'SpendingScore']
print(df.head())
print(f"\nShape: {df.shape}")
 
# ── 2. Feature selection ──────────────────────────────────────────────────────
# Primary: Income vs Spending Score (most insightful for mall segmentation)
X_2d = df[['Income', 'SpendingScore']].values
scaler_2d = StandardScaler() # line is added to scale x_2d
X_2d_scaled = scaler_2d.fit_transform(X_2d) #Added this line to scale the 2d data, not only x_full
 
# Full feature set (Age + Income + SpendingScore), scaled
X_full = df[['Age', 'Income', 'SpendingScore']].values
scaler_full = StandardScaler() #name changed from scaler to scaler_full
X_full_scaled = scaler_full.fit_transform(X_full) #changed the name for clarification since two different scaled variables are now in code
 
# ── 3. Elbow Method ───────────────────────────────────────────────────────────
inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_2d_scaled) #Changed from X_2d to X_2d_scaled to apply scaled data for elbow calculation
    inertias.append(km.inertia_)

#Added the following code to make sure the elbow method is actually used and detecting the elbow using the kneedle method
x = np.array(list(k_range), dtype=float)
y = np.array(inertias, dtype=float)

p1 = np.array([x[0], y[0]])
p2 = np.array([x[-1], y[-1]])
line_vec = p2 - p1
distances = []
for i in range(len(x)):
    point = np.array([x[i], y[i]])
    cross = np.cross(line_vec, p1 - point)
    dist = np.abs(cross) / np.linalg.norm(line_vec)
    distances.append(dist)
K_OPTIMAL = int(x[np.argmax(distances)])
    
# ── 4. Fit final model ────────────
km_2d = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df['Cluster_2D'] = km_2d.fit_predict(X_2d_scaled) #Changed from X_2d to X_2d_scaled to apply scaled data for final model
 
km_full = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df['Cluster_Full'] = km_full.fit_predict(X_full_scaled) #Changed X_scaled to X_full_scaled due to name change 
 
# PCA for 3-feature visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_full_scaled) #Changed X_scaled to X_full_scaled due to name change 
 
# ── 5. Cluster labels (manually assigned after inspection) ────────────────────
CLUSTER_NAMES = {
    0: 'Cluster 0',
    1: 'Cluster 1',
    2: 'Cluster 2',
    3: 'Cluster 3',
    4: 'Cluster 4',
}
 
PALETTE = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
 
# ── 6. Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0F1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
 
ax_elbow   = fig.add_subplot(gs[0, 0])
ax_main    = fig.add_subplot(gs[0, 1:])
ax_pca     = fig.add_subplot(gs[1, 0])
ax_age_sp  = fig.add_subplot(gs[1, 1])
ax_summary = fig.add_subplot(gs[1, 2])
 
def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor('#1A1D27')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color='#AAAAAA', fontsize=9)
    ax.set_ylabel(ylabel, color='#AAAAAA', fontsize=9)
    ax.tick_params(colors='#AAAAAA', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')
    ax.grid(color='#2A2D3A', linestyle='--', linewidth=0.5, alpha=0.7)
 
# — Elbow curve ----------------------------------------------------------------
style_ax(ax_elbow, 'Elbow Method', 'Number of Clusters (k)', 'Inertia (WCSS)')
ax_elbow.plot(k_range, inertias, 'o-', color='#3498DB', linewidth=2, markersize=6, markerfacecolor='white')
ax_elbow.axvline(K_OPTIMAL, color='#E74C3C', linestyle='--', linewidth=1.5, label=f'k = {K_OPTIMAL}')
ax_elbow.legend(fontsize=8, labelcolor='white', facecolor='#1A1D27', edgecolor='#333344')
 
# — Main: Income vs Spending Score --------------------------------------------
style_ax(ax_main, 'K-Means Clusters — Annual Income vs Spending Score', 'Annual Income (k$)', 'Spending Score (1–100)')
for c in range(K_OPTIMAL):
    mask = df['Cluster_2D'] == c
    ax_main.scatter(df.loc[mask, 'Income'], df.loc[mask, 'SpendingScore'],
                    s=80, color=PALETTE[c], alpha=0.85, edgecolors='white', linewidths=0.4,
                    label=CLUSTER_NAMES[c])
# Centroids

centroids_scaled = km_2d.cluster_centers_
centroids_original = scaler_2d.inverse_transform(centroids_scaled)
ax_main.scatter(centroids_original[:, 0], centroids_original[:, 1], s=220, marker='*', color='white',
                edgecolors='#FFD700', linewidths=1.2, zorder=5, label='Centroids')
ax_main.legend(fontsize=8.5, labelcolor='white', facecolor='#1A1D27',
               edgecolor='#333344', loc='upper left')
 
# — PCA plot ------------------------------------------------------------------
style_ax(ax_pca, 'PCA — Age + Income + Spending Score', f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
for c in range(K_OPTIMAL):
    mask = df['Cluster_Full'] == c
    ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=55, color=PALETTE[c], alpha=0.8, edgecolors='white', linewidths=0.3,
                   label=f'C{c}')
ax_pca.legend(fontsize=7.5, labelcolor='white', facecolor='#1A1D27', edgecolor='#333344')
 
# — Age vs Spending Score -----------------------------------------------------
style_ax(ax_age_sp, 'Age vs Spending Score', 'Age', 'Spending Score (1–100)')
for c in range(K_OPTIMAL):
    mask = df['Cluster_2D'] == c
    ax_age_sp.scatter(df.loc[mask, 'Age'], df.loc[mask, 'SpendingScore'],
                      s=60, color=PALETTE[c], alpha=0.8, edgecolors='white', linewidths=0.3)
 
# — Cluster summary bar -------------------------------------------------------
style_ax(ax_summary, 'Cluster Profiles (Mean Values)', 'Cluster', 'Mean Value')
summary = df.groupby('Cluster_2D')[['Age', 'Income', 'SpendingScore']].mean()
x = np.arange(K_OPTIMAL)
w = 0.26
bars_age  = ax_summary.bar(x - w,   summary['Age'],          width=w, color='#3498DB', alpha=0.85, label='Age')
bars_inc  = ax_summary.bar(x,       summary['Income'],        width=w, color='#2ECC71', alpha=0.85, label='Income (k$)')
bars_sp   = ax_summary.bar(x + w,   summary['SpendingScore'], width=w, color='#E74C3C', alpha=0.85, label='Spending Score')
ax_summary.set_xticks(x)
ax_summary.set_xticklabels([f'C{i}' for i in range(K_OPTIMAL)], color='#AAAAAA')
ax_summary.legend(fontsize=7.5, labelcolor='white', facecolor='#1A1D27', edgecolor='#333344')
 
# — Title ---------------------------------------------------------------------
fig.suptitle('Mall Customer Segmentation — K-Means Clustering (k=5)',
             color='white', fontsize=15, fontweight='bold', y=0.98)
 
plt.savefig('kmeans_clusters_v3.png', dpi=150, #Just changed the name of the saved png to clarify differences between version outputs
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("\n✅ Plot saved as kmeans_clusters.png")
 
# ── 7. Console summary ────────────────────────────────────────────────────────
print("\n── Cluster Summary (Income vs Spending Score features) ──")
print(df.groupby('Cluster_2D')[['Age','Income','SpendingScore']].mean().round(1))
print("\n── Cluster Sizes ──")
print(df['Cluster_2D'].value_counts().sort_index())