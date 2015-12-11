import BayesBimodalTest as BBT
import matplotlib.pyplot as plt
import numpy as np

N = 1000
dataA = np.random.normal(3, 1, N/2)
dataB = np.random.normal(-3, 1, N/2)
data = np.concatenate([dataA, dataB])
test = BBT.BayesBimodalTest(data, nburn0 = 100, nburn=200, nprod=300, ntemps=1, nwalkers=100)
