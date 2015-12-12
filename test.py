import BayesBimodalTest as BBT
import matplotlib.pyplot as plt
import numpy as np

N = 5000
dataA = np.random.normal(2, 1, N/2)
dataB = np.random.normal(-2, 1, N/2)
data = np.concatenate([dataA, dataB])
test = BBT.BayesBimodalTest(data, nburn0=100, nburn=100, nprod=100, ntemps=50)
test.BayesFactor()
