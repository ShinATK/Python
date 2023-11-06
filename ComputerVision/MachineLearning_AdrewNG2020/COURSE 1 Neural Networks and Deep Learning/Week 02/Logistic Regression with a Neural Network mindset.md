
- 小技巧：如果想要将一个(a, b, c, d)大小的矩阵X，拉平为(b * c * d, a)时
```python
X_flatten = X.reshape(X.shape[0], -1).T
```

**What you need to remeber:**
	Common steps for pre-processing a new dataset are:
- Figure out the dimensions and shapes of the problem (m_train, num_px, ... )
- Reshape the datasets such that each examples is now a vector of size (num_px * num_px * 3, 1)
- "Standardize" the data