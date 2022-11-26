import numpy as np
import matplotlib.pyplot as plt
import neural_network
import parameters as params

def check_metropolis_sampling():
    #target_p_func = lambda x: neural_network.calc_exact_psi(x)**2
    target_p_func = lambda x: neural_network.calc_train_psi(x)**2
    L = 5.0
    sample_nums = [10**n for n in [2,3,4,5]]
    xlinspace = np.linspace(-L/2, L/2, 1000, dtype=float)
    target_p = np.vectorize(target_p_func)(xlinspace)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Check Metropolis Sampling")
    for i, sample_num in enumerate(sample_nums):
        xs = neural_network.metropolis(target_p_func, randomwalk=False, sample_n=sample_num)
        ax = fig.add_subplot(2,2,i+1)
        ax.hist(xs, bins=int(np.sqrt(len(xs))),density=True)
        ax.plot(xlinspace, target_p)
        ax.set_title(f"SampleN : {sample_num}")
        ax.set_xlabel("x")
        ax.set_ylabel("properbility")
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    exit()

