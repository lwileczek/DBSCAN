---?color=white
@title[Introduction]

# DBSCAN
#### Density-based spatial clustering of applications with noise

@snap[south-east span-20]
@size[0.75em](Luke Wileczek 2019-04-20)
@snapend

---
@title[clustering methods]

#### Can you name a clustering method?

<br>

@css[fragment](Did you say K-Means?)

+++
@title[K-Means]

## K-Means

@snap[text-left]
Why I don't like K-Means:
@snapend

  - Only works on [convex sets](https://en.wikipedia.org/wiki/Convex_set)
  - Even on perfect data sets, it can get stuck in a local minimum
  - Unevenly Sized Clusters
  - Fixed number of clusters, Always K cluster regardless of the truth
  - Everything [in here](https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means)

---
@title[What is DBSCAN]

### What is DBSCAN?

<br>

@snap[west text-left span-50]
DBSCAN is an unsupervised density-based clustering algorithm, meaning that 
the algorithm focuses on the distance between each point and the count of neighbors rather than the distance to a centroid like K-Means.
@snapend

@snap[east span-50]
![](/assets/img/explanation.png)
@snapend

+++
@title[Fromal definition]

Given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away). [Wiki](https://en.wikipedia.org/wiki/DBSCAN)

---
@title[intuition]

## Intuition
Let's build some Intuition before we start the code  
https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/ 

---
@title[Overview]

## Algorithm Overview
When constructing the DBSCAN algorithm we need four things:

@ul
  - Minimum amount of points (minPts)
  - Distance function (dist\_func)
  - Distance from each point (Eps)
  - data
@ulend

Note:
To compute DBSCAN we'll go point by point and check if it is a core point 
or not. 
If it is a core point we will create a new cluster, then search through 
all of its neighbors. 
We'll add the neighborhoods of all these points to the cluster. 
If one of the points is a core point as well, it's neighborhood will be 
added to our search. 
This will continue until we cannot reach anymore points. 
Then we'll move on to the next point that we haven't visited/labeled yet. 
Approaching the algorithm this way instead of calculating all of the distances 
at once will help to save memory.

+++
@title[Fun Fact 0]

### Fun fact time

Eps stands for epsilon, the Greek letter, which is used in mathematics to represent a region around an object. An arbitrary but common choice.


--- 
@title[init Algorithm]

## Helper Functions

@[1-5]
@[7-11]

```python
def euclidean(x, y):
    """
    Calculate the euclidean distance between two vectors
    """
    return np.sqrt(np.sum((x-y)**2))

def find_neighbors(db, dist_func2, p, e):
    """
    return the indecies of all points within epsilon of p
    """
    return [idx for idx, q in enumerate(db) if dist_func2(p,q) <= e]
```

+++
@title[main function]

```
def dbscan(data, min_pts, eps, dist_func=euclidean):
    """
    Run the DBSCAN clustering algorithm
    """
    C = 0                          # cluster counter
    labels = {}                    # Dictionary to hold all of the clusters
    visited = np.zeros(len(data))  # check to see if we visited this point
```

Note:
inital values and set default distance

+++

@title[for-loop]

```python
for idx, point in enumerate(data):
    # If already seached this point move on
    if visited[idx] == 1:
        continue
    visited[idx] = 1  # have now visited this point
    # all of point P's neighbors
    neighbors = find_neighbors(data, dist_func, point, eps)
```

+++

@title[if/then]

```python
if len(neighbors) < min_pts:
            labels.setdefault('noise', []).append(idx)
        else:
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

