import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from gensim.models import Word2Vec
import random

# --- Configuration for Cleaner Signal ---
N = 1500
k_communities = 2  # Keep it simple (2 balanced blocks)
window_size = 10
dim = 16
walk_length = 80   # Longer walks = less sampling noise
num_walks = 10

def generate_balanced_dcsbm(n):
    """Generates a Balanced 2-Block DCSBM to minimize community variance"""
    # 1. Degrees: Power law but slightly cleaner range
    degrees = np.random.zipf(2.5, n).astype(float)
    degrees = np.clip(degrees, 10, 150)
    theta = degrees / degrees.sum()
    
    # 2. Balanced Communities
    z = np.zeros(n, dtype=int)
    z[n//2:] = 1  # Half and half
    
    # 3. Symmetric Block Matrix
    # Mixing parameter: Strong internal (0.1), Weak external (0.005)
    B = np.array([[0.1, 0.005], 
                  [0.005, 0.1]])
    
    # 4. Generate Adjacency
    P = np.outer(theta, theta) * 5000 # Scaling constant
    
    # Apply mask
    mask = np.zeros((n, n))
    mask[:n//2, :n//2] = B[0,0]
    mask[n//2:, n//2:] = B[1,1]
    mask[:n//2, n//2:] = B[0,1]
    mask[n//2:, :n//2] = B[1,0]
    
    P = P * mask
    A = (np.random.rand(n, n) < P).astype(float)
    A = np.triu(A, 1) + np.triu(A, 1).T
    
    # Largest component
    G = nx.from_numpy_array(A)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    
    # Get final matrices
    A = nx.to_numpy_array(G)
    d = A.sum(axis=1)
    z = z[list(G.nodes())] # Update labels for subgraph
    
    return A, d, G, z

# --- Run ---
print("1. Generating Balanced DCSBM...")
A, d, G, communities = generate_balanced_dcsbm(N)

print("2. Computing Spectral Theory...")
# Normalized Laplacian
inv_sqrt_d = 1.0 / np.sqrt(d)
D_inv_sqrt = np.diag(inv_sqrt_d)
S = D_inv_sqrt @ A @ D_inv_sqrt

# Get Top-2 Eigenvectors
# v0 is the stationary distribution (should be exactly sqrt(d))
# v1 is the Fiedler vector (community signal)
evals, evecs = eigsh(S, k=2, which='LA')

# Theory Norm: We look at v1 (Fiedler) which contains the community split
# In DCSBM, v1_i should approx equal sqrt(theta_i) * sign(community)
u_theory = evecs[:, 0] # v1 (index 0 in eigsh usually smallest? No 'LA' is largest algebraic)
# Check if v0 or v1 is the stationary one (all positive). 
# Usually v0 is all positive. Let's find the one correlated with sqrt(d)
corrs = [np.corrcoef(np.abs(evecs[:,i]), np.sqrt(d))[0,1] for i in range(2)]
print(f"   Eigenvector correlations with sqrt(d): {corrs}")
# We pick the one that carries the community signal (usually the 2nd one, but let's look at the norm of the subspace)
theory_norm = np.abs(evecs[:, 1]) # Use the Fiedler vector magnitude

print("3. Training Node2Vec...")
# Standard gensim setup
walks = []
nodes = list(G.nodes())
for _ in range(num_walks):
    random.shuffle(nodes)
    for node in nodes:
        path = [str(node)]
        for _ in range(walk_length-1):
            nbrs = list(G.neighbors(int(path[-1])))
            if not nbrs: break
            path.append(str(random.choice(nbrs)))
        walks.append(path)

model = Word2Vec(sentences=walks, vector_size=dim, window=window_size, 
                 min_count=0, sg=1, negative=5, workers=4, epochs=10)

embedding_matrix = np.zeros((len(G.nodes()), dim))
for i in range(len(G.nodes())):
    if str(i) in model.wv:
        embedding_matrix[i] = model.wv[str(i)]
        
n2v_norm = np.linalg.norm(embedding_matrix, axis=1)

# --- Plotting ---
plt.figure(figsize=(12, 6))

# Plot A: Theory (Spectral)
plt.subplot(1, 2, 1)
plt.scatter(d, theory_norm, c=communities, cmap='coolwarm', alpha=0.6, s=15)
# Fit line
z = np.polyfit(d, theory_norm, 1)
plt.plot(d, np.poly1d(z)(d), 'k--', lw=1, alpha=0.5, label='Trend')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Degree (log)')
plt.ylabel('|Eigenvector Entry| (log)')
plt.title('Theory: Laplacian Eigenvector (~ sqrt(d))')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot B: Empirical (Node2Vec)
plt.subplot(1, 2, 2)
plt.scatter(d, n2v_norm, c=communities, cmap='coolwarm', alpha=0.6, s=15)
z = np.polyfit(np.log(d), np.log(n2v_norm+1e-9), 1) # Log-log fit
plt.plot(d, np.exp(np.poly1d(z)(np.log(d))), 'k--', lw=1, alpha=0.5, label='Trend')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Degree (log)')
plt.ylabel('Embedding Norm (log)')
plt.title('Reality: Node2Vec Norms (~ 1/d)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('corrected_plots.png')
print("Saved corrected_plots.png")