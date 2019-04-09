import numpy as np

def euclidean(x,y):
    """
    Calculate the euclidean distance between two vectors
    """
    return np.sqrt(np.sum( (x-y)**2))


def find_neighbors(db, dist_func2, p, e):
    """
    return the indecies of all points within epsilon of p
    """
    return  [idx for idx, q in enumerate(db) if dist_func2(p,q) <= e]

def dbscan(data, min_pts, eps, dist_func=euclidean):
    """
    Run the DBSCAN clustering algorithm
    """
    C = 0                          # cluster counter
    labels = {}                    # Dictionary to hold all of the clusters
    visited = np.zeros(len(data))  # check to see if we visited this point

    # going point by point through our dataset find the neighborhood and
    # determine if it is a core point. if it is, then search through all of its
    # neighbors and add them to the cluster that core point is in. Do this
    # until all points have been visited.
    for idx, point in enumerate(data):
        if visited[idx] == 1:
            continue
        visited[idx] = 1
        # all of P's neighbors
        neighbors = find_neighbors(data, dist_func, point, eps)

        # if it is not a core point then all points in its neighborhood are
        # boundary points. Thus, it is noise or an outlier.
        if len(neighbors) < min_pts:
            labels.setdefault('noise', []).append(idx)
        else:
            # else, we have a new cluster. Search through all points reachable
            # and add them to this cluster.
            C += 1
            # We want to use a list because it will extend the search
            labels.setdefault(C, []).append(idx)
            neighbors.remove(idx)  # Already been checked. Will be added below
            for q in neighbors:
                if visited[q] == 1:
                    continue
                visited[q] = 1
                q_neighbors = find_neighbors(data, dist_func, data[q, :], eps)
                if len(q_neighbors) >= min_pts:
                    neighbors.extend(q_neighbors)  # extend the search
                labels[C].append(q)

    return labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    # make sample dataset
    X, y = make_moons(n_samples=750, shuffle=True, noise=0.11, random_state=42)
    X = StandardScaler().fit_transform(X)
    # run algorithm
    data_labels = dbscan(X, 4, 0.3, dist_func=euclidean)
    # plot resutls
    for key,values in data_labels.items():
        plt.plot(X[values][:, 0], X[values][:, 1], 'o', markersize=6, label=key)
    plt.show()
