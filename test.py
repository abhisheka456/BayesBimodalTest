import BayesBimodalTest as BBT
import numpy as np

N = 5000
dataA = np.random.normal(2, 1.5, N/2)
dataB = np.random.normal(-2, 1.5, N/2)
data = np.concatenate([dataA, dataB])
test = BBT.BayesBimodalTest(data, [1, 2], nburn0=100, nburn=100, nprod=100,
                            ntemps=50)
test.diagnostic_plot()
test.BayesFactor()
