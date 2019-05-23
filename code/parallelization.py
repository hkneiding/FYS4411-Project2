from tests import run_tests
import os
import multiprocessing as mp
import numpy as np
import time #and space

start_time = time.time()

""" Parallell code for running through different parameters. """


n_l = 18
n_t = 3

learning_rate = np.linspace(0.01,1.0, n_l)
time_step = np.linspace(0.1, 1.0, n_t)

np.save("results/counter.npy", 0)
iteration = 0
for rate in learning_rate:
    for step in time_step:
        p = mp.Process(target=run_tests, args=(rate, step, iteration))
        p.start()
        iteration +=1
p.join()


print("Finished!")

end_time = time.time()
print("Time spent: ", end_time - start_time, " seconds.")
print("CPU time: ", (end_time-start_time)*8)
