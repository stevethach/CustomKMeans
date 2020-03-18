import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
# Flat clustering, k means algorithm
import numpy as np
from sklearn.datasets.samples_generator import make_blobs


colors = 10*["g","b","c","r","k","o"]

X, y = make_blobs(n_samples=15, centers=3, n_features=2)

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self,data):
        self.centroids = {}
        
        for i in range(self.k):
            # k, is number of clusters
            # start centroid guess i = 0 at [1,2]
            # second centroid guess i = 1 at [1.5,1.8]
            self.centroids[i] = data[i]
        
        for i in range(self.max_iter):
            
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
        
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
              
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            
            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
                    print()
                    optimized = False
                    
            if optimized:
                break
                
            
    
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
        
clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker = "o", color = "k", s=150, linewidths=5)
    
for classification in clf.classifications:
    color1 = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker="x",color=color1, s=150, linewidths = 5)
        

plt.show()