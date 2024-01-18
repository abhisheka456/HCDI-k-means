# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
cimport numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import scipy
import math
import sys


cdef extern from "mutual_neighborhood_graph.h":
    void compute_mutual_knn(int n, int k, int d,
                    double * radii,
                    int * neighbors,
                    double beta,
                    double epsilon,
                    int * result)
    void cluster_remaining(int n, int k, int d,
                   double * dataset,
                   double * radii,
                   int * neighbors,
                   int * initial_memberships,
                   int * result)
    void calculate_centre(int n,
                      int d,
                      int ns_cluster,
                      int * membership,
                      double * X,
                      double * ns_centre)


cdef compute_mutual_knn_np(n, k, d,
                    np.ndarray[double,  ndim=1, mode="c"] radii,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] neighbors,
                    beta,
                    epsilon,
                    np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    compute_mutual_knn(n, k, d,
                        <double *> np.PyArray_DATA(radii),
                        <int *> np.PyArray_DATA(neighbors),
                        beta, epsilon,
                        <int *> np.PyArray_DATA(result))

cdef cluster_remaining_np(n, k, d,
                    np.ndarray[double,  ndim=2, mode="c"] dataset,
                    np.ndarray[double,  ndim=1, mode="c"] radii,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] neighbors,
                    np.ndarray[np.int32_t, ndim=1, mode="c"] initial_memberships,
                    np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining(n, k, d,
                    <double *> np.PyArray_DATA(dataset),
                    <double *> np.PyArray_DATA(radii),
                    <int *> np.PyArray_DATA(neighbors),
                    <int *> np.PyArray_DATA(initial_memberships),
                    <int *> np.PyArray_DATA(result))

cdef calculate_centre_np(n, d, ns_cluster,
                         np.ndarray[np.int32_t, ndim=1, mode="c"] membership,
                         np.ndarray[double,  ndim=2, mode="c"] X,
                         np.ndarray[double,  ndim=2, mode="c"] ns_centre):
    calculate_centre(n, d, ns_cluster,
                     <int *> np.PyArray_DATA(membership),
                     <double *> np.PyArray_DATA(X),
                     <double *> np.PyArray_DATA(ns_centre))

class DIKMeans:
    """
    Parameters
    ----------
    
    k: The number of neighbors (i.e. the k in k-NN density)

    beta: Ranges from 0 to 1. We choose points that have kernel density of at
        least (1 - beta) * F where F is the mode of the empirical density of
        the cluster

    epsilon: For pruning. Sets how much deeper in the cluster tree to look
        in order to connect clusters together. Must be at least 0.


    Attributes
    ----------

    n_clusters: number of clusters fitted

    cluster_map: a map from the cluster (zero-based indexed) to the list of points
        in that cluster

    """



    def __init__(self, k, beta,
                    epsilon=0,
                    ann="kd_tree"):
        self.k = k
        self.beta = beta
        self.epsilon = epsilon
        self.ann = ann



    def fit(self, X):
        """
        Determines the clusters in two steps.
        First step is to compute the knn density estimate and
        distances. This is done using kd tree
        Second step is to build the knn neighbor graphs
        Updates the cluster count and membership attributes

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint in 
            euclidean space
        """
        X = np.array(X)
        n, d = X.shape
        knn_density = None
        neighbors = None
        
        
        neigh = NearestNeighbors(n_neighbors=self.k, algorithm=self.ann)
        neigh.fit(X)
        query_res = neigh.kneighbors(X)
        knn_radius = query_res[0][:, self.k-1]
        neighbors = query_res[1]

        

        memberships = np.zeros(n, dtype=np.int32)
        result = np.zeros(n, dtype=np.int32)
        neighbors = np.ndarray.astype(neighbors, dtype=np.int32)
        knn_radius = np.ndarray.astype(knn_radius, dtype=np.float64)
        X_copy = np.ndarray.astype(X, dtype=np.float64)

        compute_mutual_knn_np(n, self.k, d,
                            knn_radius,
                            neighbors,
                            self.beta, self.epsilon,
                            memberships)
        knn_radius = np.ndarray.astype(knn_radius, dtype=np.float64)
        cluster_remaining_np(n, self.k, d, X_copy, knn_radius, neighbors, memberships, result)
        n_cluster = np.amax(result)
        n_centers = np.zeros((n_cluster+1, d))
        X_copy = np.ndarray.astype(X, dtype=np.float64)

        calculate_centre_np(n, d, n_cluster+1, result, X_copy, n_centers)   
        
        Algo = KMeans(n_clusters = n_cluster+1, init = n_centers, n_init = 1)
        Algo.fit(X_copy)
        result = Algo.labels_.astype(int)
        #print(n_centers)
        self.memberships = result
        self.ns_centers = n_centers


