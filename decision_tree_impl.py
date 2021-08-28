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
