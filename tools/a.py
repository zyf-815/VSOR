import pandas as pd

gt_index = pd.Series([1])
rank_index = pd.Series([2])

p = gt_index.corr(rank_index, method='pearson')

print(p)