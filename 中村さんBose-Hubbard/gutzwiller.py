import numpy as np
from parameters import *

def normalize(phi):
    norm2 = (phi**2).sum()
    phi /= np.sqrt(norm2)
    return norm2

def update(data):
    phi = data["phi"]
    beta = data["beta"] = (N_SQRT[1:] * phi[:-1] * phi[1:]).sum()
    data["E"] = -2 * data["J"] * beta**2 
    data["E"] = (- data["mu"]*phi**2 + 0.5 * (N_SEQ**2 - N_SEQ) * phi**2 ).sum()

    phi_p = np.append((N_SQRT * phi)[1:], [0.0])
    phi_m = N_SQRT * np.append([0], phi[:-1])
    dEdPhi = -4 * data["J"] * beta * (phi_p + phi_m)
    dEdPhi += - 2*data["mu"]* N_SEQ * phi
    dEdPhi += N_SEQ * (N_SEQ - 1) * phi
    phi -= ETA * dEdPhi
    normalize(phi)

def calc(mu, J, data):
    if data==None:
        data = {}
        data["phi"] = np.exp(-np.array(range(N_P)))
        data["beta"] = None
        data["E"] = None
        normalize(data["phi"])
    data["mu"] = mu
    data["J"] = J
    data["OK"] = False

    Eold = 1
    for t in range(ITER_MAX):
        update(data)
        if np.abs(data["E"]-Eold) < EPS:
            data["OK"] = True
            return data
        Eold = data["E"]
    return data
