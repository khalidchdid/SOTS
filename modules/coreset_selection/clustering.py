import numpy as np
from sklearn.cluster import KMeans


class KMeansCoresetSelector:
   """
   Coreset selection via k-means clustering in the signature space R^27.

   Runs k-means with k clusters, then selects the point closest to each
   centroid (medoid) as the representative of that cluster.

   This is equivalent to the k-Center / Herding baseline commonly used
   in the coreset selection literature (DeepCore, GradMatch, etc.).

   Args:
       X          : np.array of shape (n, d), full dataset of signature points
       random_state : random seed for reproducibility (default: 42)

   Example:
       selector = KMeansCoresetSelector(X)
       selected_idx, cluster_labels = selector.select(k=200)
   """

   def __init__(self, X: np.ndarray, random_state: int = 42):
       self.X = X
       self.n, self.d = X.shape
       self.random_state = random_state

   def select(
       self,
       k: int,
       verbose: bool = True,
   ) -> tuple:
       """
       Select a coreset of size k via k-means medoids.

       Args:
           k       : number of clusters = number of selected points
           verbose : print progress
       Returns:
           selected_idx   : np.array of shape (k,), indices into X
           cluster_labels : np.array of shape (n,), cluster assignment
                            for each point in X
       """
       if k > self.n:
           raise ValueError(f"k={k} exceeds dataset size n={self.n}")

       if verbose:
           print(f"Running k-means with k={k} clusters on {self.n} points "
                 f"in R^{self.d}...")

       # Step 1 — k-means clustering
       kmeans = KMeans(
           n_clusters=k,
           random_state=self.random_state,
           n_init=10,
       )
       cluster_labels = kmeans.fit_predict(self.X)
       centroids = kmeans.cluster_centers_   # (k, d)

       if verbose:
           print(f"k-means converged. Selecting medoids...")

       # Step 2 — For each cluster, select the point closest to centroid
       selected_idx = np.zeros(k, dtype=int)

       for c in range(k):
           # Indices of points in cluster c
           cluster_mask = np.where(cluster_labels == c)[0]

           if len(cluster_mask) == 0:
               # Empty cluster (rare) — pick a random point
               selected_idx[c] = np.random.randint(0, self.n)
               continue

           # Points in cluster c
           cluster_points = self.X[cluster_mask]   # (n_c, d)

           # Distance to centroid
           dists = np.linalg.norm(cluster_points - centroids[c], axis=1)

           # Medoid = closest point to centroid
           local_best = np.argmin(dists)
           selected_idx[c] = cluster_mask[local_best]

       if verbose:
           print(f"Selected {k} medoids.")

       return selected_idx, cluster_labels