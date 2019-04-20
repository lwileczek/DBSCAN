# DBSCAN
Example of DBSCAN; what it is, how to use it and why. 

## What is DBSCAN?
Is an unsupervised density-based clustering algorithm. Density-based means that 
the algorithm focuses on the distance between each point and it's neighbors 
instead of the distance to a centroid like K-Means. One way to describe DBSCAN 
is: 
> given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away). [Wiki](https://en.wikipedia.org/wiki/DBSCAN)

One thing that is interesting is that we can define what _close_ means by using
 different distance functions. For this tutorial we'll use [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance), 
which is basically the normal way we measure distance. 
Another way to say how DBSCAN works is it finds central or core points and finds
which points are connected to those core points. All points reachable by the 
core points are clustered together. A point is a core point if it has some minimum 
amount of neighbors within a given radius. A point _q_ is reachable from _p_ if 
there is a path p<sub>1</sub>, ..., p<sub>n</sub> with p<sub>1</sub> = p and 
p<sub>n</sub> = q, 
where each p<sub>i+1</sub> is directly reachable from p<sub>i</sub> (all the 
points on the path must be core points, with the possible exception of q).
The following visual shows core points __A__ with reachable points 
__B__ & __C__. The point __N__ is considered an outlier. 

<div style="width:402px; font-size: 94%; text-align: center;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/af/DBSCAN-Illustration.svg" alt="wikimedia_DBSCAN" style="width:400; height: 288;">
</div>

You may be wondering why
 we care since everyone already knows K-Means. 
 Well, The DBSCAN algorithm views clusters as areas of high density separated 
 by areas of low density. 
 Due to this rather generic view, 
clusters found by DBSCAN can be any shape, as opposed to k-means which 
assumes that clusters are convex shaped. 
So we gain a little bit more flexibility. Below we'll see some examples of how 
DBSCAN outperforms K-Means.

## Algorithm Overview
When constructing the DBSCAN algorithm we need four things:
  1. Minimum amount of points (minPts)
  2. Distance function (dist\_func)
  3. Distance from each point (Eps)
  4. data

> ASIDE: Eps stands for epsilon, the Greek letter, which is used in mathematics
> to represent a region around an object. An arbitrary but common choice. 

To compute DBSCAN we'll go point by point and check if it is a core point or not. 
If it is a core point we will create a new cluster, then search through all of its 
neighbors. We'll add the neighborhoods of all these points to the cluster. If one of 
the points is a core point as well, it's neighborhood will be added to our search. 
This will continue until we cannot reach anymore points. Then we'll move on to the 
next point that we haven't visited/labeled yet. Approaching the algorithm this way 
instead of calculating all of the distances at once will help to save memory. 


## Writing the algorithm
We'll use a few packages to help us along our way. Let's start with some imports:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
```
Next we'll need some data. We are going to use Sci-Kit Learn's built in dataset
generator functions to production two half moons. We'll see what these look like
in a little bit.

```python
X, y = make_moons(n_samples=750, shuffle=True, noise=0.11, random_state=42)
X = StandardScaler().fit_transform(X)
```

Now before we write out DBSCAN function we'll need two helper functions. The
first is our distance function and the second will collect the indecies of
all neighboring points for some point _p_.

```python
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
```

Now we may finally start with our DBSCAN function. We'll start with
our parameters, all of which we have seen before. _Feel free skip to 
the bottom where the entire algorithm is presented_.

```python
def dbscan(data, min_pts, eps, dist_func=euclidean):
```

Now, we'll want to initialize some variables. We'll use **C** to keep
track of how many clusters we have. We'll use a dictionary, **labels**,
to hold the clusters. Finally, we'll use an array to keep track of which
points we have already searched/visited.


```python
C = 0                          # cluster counter
labels = {}                    # Dictionary to hold all of the clusters
visited = np.zeros(len(data))  # check to see if we visited this point
```

Now we start our search through all of the data points. For each one,
we will find all of it's neighbor points and determine if it is a core
point or not. If it is not a core point we'll mark it as noise and move on.

```python
for idx, point in enumerate(data):
    if visited[idx] == 1:  # if we have seen this point already move to next
        continue
    visited[idx] = 1
    neighbors = find_neighbors(data, dist_func, point, eps)
    if len(neighbors) < min_pts:
        labels.setdefault('noise', []).append(idx)
```

If it is a core point we can create a cluster and start adding points to it.
We will add all of its neighbors to the the cluster and then one-by-one
check the neighbors of the neighbors. For each core point we come across
we will add continue the search through their neighbors until we cannot reach
another core point.

```python

C += 1
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

```
That's all the components we will need. Putting it all together we get:
```python
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
```

# Examples
We are going to test our algorithm on the following two patters: half-moons and
circle in a circle.

![sklearn.dataset.make_moons](/assets/img/raw_moons.png)

![sklearn.dataset.make_circles](/assets/img/raw_circles.png)

If we were to naively apply K-Means we wouldn't be able to predict these two groups with much satisfaction. 

![k-means moons](/assets/img/kmeans_moon.png)

![k-means circles](/assets/img/kmeans_circles.png)

K-means struggles with the moons because they are not [convex](https://en.wikipedia.org/wiki/Convex_set) which is one of 
K-means assumptions. The problem with the circles is the centroids cannot overlap and if they did it would return a single cluster. 
However, by applying DBSCAN which doesn't assume the shape of the data but proxmity between points we get the following results:

![my moons](/assets/img/my_dbscan.png)

![my moons](/assets/img/my_circles.png)

![Like a glove](https://media.giphy.com/media/YpRuexuyaJQ0U/giphy.gif)  

What a B-E-A-utiful algorithm. Lucky for us, DBSCAN has been added to many standard packages in Python, R, C# or whatever you 
code in. If you primarily use Python, Sci-Kit learn has DBSCAN built in with some efficiencies we didn't use. 

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
X, labels_true= make_moons(n_samples=750, shuffle=True, noise=0.11, random_state=42)
X = StandardScaler().fit_transform(X)
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
```
which produces:

```sh
Estimated number of clusters: 2
Homogeneity: 0.995
Completeness: 0.947
V-measure: 0.970
Adjusted Rand Index: 0.987
Adjusted Mutual Information: 0.947
Silhouette Coefficient: 0.225
```

![Sci-Kit learn DBSCAN](/assets/img/sk.png)

Sci-Kit Learn example adapted from [here](http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html).
## Caveats
Like all clustering algorithms DBSCAN comes with a few caveats. Below are some disadvantages listed on wikipedea.

  1. DBSCAN is not entirely deterministic: border points that are reachable from more than one cluster can be part of either cluster, depending on the order the data are processed. For most data sets and domains, this situation fortunately does not arise often and has little impact on the clustering result: both on core points and noise points, DBSCAN is deterministic. DBSCAN* is a variation that treats border points as noise, and this way achieves a fully deterministic result as well as a more consistent statistical interpretation of density-connected components.

  2. The quality of DBSCAN depends on the distance measure used in the function regionQuery(P,ε). The most common distance metric used is Euclidean distance. Especially for high-dimensional data, this metric can be rendered almost useless due to the so-called "Curse of dimensionality", making it difficult to find an appropriate value for ε. This effect, however, is also present in any other algorithm based on Euclidean distance.

  3. DBSCAN cannot cluster data sets well with large differences in densities, since the minPts-ε combination cannot then be chosen appropriately for all clusters.
  4. If the data and scale are not well understood, choosing a meaningful distance threshold ε can be difficult.

If you are interested in DBSCAN it may be beneficial to investigate the [OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm) 
algorithm which allows clusters to have different densities. 

## Further reading
  - [Visualizing DBSCAN](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) Very cool interactive visualization of DBSCAN
  - [OPTICS algorithm](https://en.wikipedia.org/wiki/OPTICS_algorithm)
  - [DBSCAN Wiki](https://en.wikipedia.org/wiki/DBSCAN)
  - [Sci-Kit Learn Clustering](http://scikit-learn.org/stable/modules/clustering.html#clustering)

## Bibliography
  - Figure 1 By Chire [CC BY-SA 3.0  (https://creativecommons.org/licenses/by-SA/3.0)], from Wikimedia Commons
