import numpy as np

def entropy(y):
    # assume binary classification problem
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -p0*np.log2(p0) - p1*np.log2(p1)
  
    
class TreeNode:
    
    def __init__(self, depth=1, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        if self.max_depth is not None and self.max_depth < self.depth:
            raise Exception("depth > max_depth")
            
    def fit(self, X, Y):
        if len(Y) == 1 or len(set(Y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]
            
    def find_split(self, X, Y, col):
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]
        
        # boundaries between 2 classes or midpoint between 2 points
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        for b in boundaries:
            split = (x_values[b] + x_values[b+1]) / 2
            ig = self.information_gain(x_values, y_values, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split
            
    def information_gain(self, x, y, split):
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0 #float(len(y1)) / N
        return entropy(y) - p0*entropy(y0) - p1*entropy(y1)
