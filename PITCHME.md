---?color=white
@title[Introduction]

# DBSCAN
#### Density-based spatial clustering of applications with noise

@snap[south-east]
@size[2em](Luke Wileczek 2019-04-20)
@snapend

---
@title[clustering methods]

#### Can you name a clustering method?

<br>

@css[fragment](Did you say K-Means?)

+++
@title[K-Means]

### K-Means
K-Means is generally the first clustering method people learn. However, K-Means
has a lot of properties that can be undesirable.

  - Only works on [convex sets](https://en.wikipedia.org/wiki/Convex_set)
  - Even on perfect data sets, it can get stuck in a local minimum
  - Unevenly Sized Clusters
  - Fixed number of clusters, Always K cluster regardless of the truth
  - Everything [in
    here](https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means)

---
@title[What is DBSCAN]

### What is DBSCAN?

@snap[west]
DBSCAN is an unsupervised density-based clustering algorithm, meaning that 
the algorithm focuses on the distance between each point and the count of neighbors rather than the distance to a centroid like K-Means.
@snapend

@snap[east]
![](assets/img/explination.png)
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
  - Distance function (dist_func)
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

Yeah boi
