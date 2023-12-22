import netket as nk
from netket.operator.spin import sigmax,sigmaz, identity, sigmam, sigmap
import jax
import pickle
import matplotlib.pyplot as plt
from netket.graph import Graph, Chain
from netket.operator._local_operator import LocalOperator
import numpy as np
from jax import numpy as jnp
from scipy.sparse.linalg import eigsh

def to_array(model, parameters, hilbert):
    # begin by generating all configurations in the hilbert space.
    # all_States returns a batch of configurations that is (hi.n_states, N) large.
    all_configurations = hilbert.all_states()

    # now evaluate the model, and convert to a normalised wavefunction.
    logpsi = model.apply(parameters, all_configurations)
    psi = jnp.exp(logpsi)
    psi = psi / jnp.linalg.norm(psi)
    return logpsi

def proj(hilbert,site, state:int):
    idx = state+1
    proj_mat = np.zeros((hilbert.local_size,hilbert.local_size))
    proj_mat[idx,idx] = 1
    return LocalOperator(hilbert, acting_on=[site], operators= proj_mat)

def create_ham(m, N, spin_1_hilbert,t=1, l_cons = 30):
    H = 0
    for i in range(1,N-1):
        H += -t/2*sigmap(spin_1_hilbert,i)*sigmam(spin_1_hilbert,(i+1)%N)*proj(spin_1_hilbert,i,0)*proj(spin_1_hilbert,(i+1)%N,0)
        H+= -t/2*sigmam(spin_1_hilbert,i)*sigmap(spin_1_hilbert,(i+1)%N)*proj(spin_1_hilbert,i,1)*proj(spin_1_hilbert,(i+1)%N,-1)
    # for i in range(N):
    #     H+= (sigmaz(spin_1_hilbert,i)*(1/2) +1 -sigmaz(spin_1_hilbert,i)*sigmaz(spin_1_hilbert,i)*(1/4) - (-1)**i*1/2)*(sigmaz(spin_1_hilbert,i)*(1/2) +1 -sigmaz(spin_1_hilbert,i)*sigmaz(spin_1_hilbert,i)*(1/4) - (-1)**i*1/2)
    #     H += -m*sigmaz(spin_1_hilbert,i)*sigmaz(spin_1_hilbert,i)*(1/4)
    for i in range(N):
        H+= ((-1)**i*(-sigmaz(spin_1_hilbert,i)*(1/2) +1 -sigmaz(spin_1_hilbert,i)*sigmaz(spin_1_hilbert,i)*(1/4)) - (-1)**i*1/2)*((-1)**i*(sigmaz(spin_1_hilbert,i)*(1/2) +1 -sigmaz(spin_1_hilbert,i)*sigmaz(spin_1_hilbert,i)*(1/4)) - (-1)**i*1/2)
        H += -m*sigmaz(spin_1_hilbert,i)*sigmaz(spin_1_hilbert,i)*(1/4)
    H += l_cons*((sigmaz(spin_1_hilbert,0) -2)*(sigmaz(spin_1_hilbert,0)*(1/2)-1) + (sigmaz(spin_1_hilbert,N-1)*(1/2)+1)*(sigmaz(spin_1_hilbert,N-1)*(1/2)+1))
    H += H.conjugate().transpose()
    id = identity(spin_1_hilbert)
    return H

def create_local_ham(i,m, N, spin_1_hilbert,t=1, l_cons = 30, return_type= "dense"):
    #returns local Hamiltonian of the system as dense matrices (default) for TEBD. For argument i, return H_{i,i+1}
    H = 0
    H+= (sigmaz(spin_1_hilbert,i)*(1/2) +1 -sigmaz(spin_1_hilbert,i)*sigmaz(spin_1_hilbert,i)*(1/4) - (-1)**i*1/2)*(sigmaz(spin_1_hilbert,i)*(1/2) +1 -sigmaz(spin_1_hilbert,i)*sigmaz(spin_1_hilbert,i)*(1/4) - (-1)**i*1/2)
    H += -m*sigmaz(spin_1_hilbert,i)*sigmaz(spin_1_hilbert,i)*(1/4)
    H += -t/2*sigmap(spin_1_hilbert,i)*sigmam(spin_1_hilbert,(i+1)%N)*proj(spin_1_hilbert,i,0)*proj(spin_1_hilbert,(i+1)%N,0)
    H+= -t/2*sigmam(spin_1_hilbert,i)*sigmap(spin_1_hilbert,(i+1)%N)*proj(spin_1_hilbert,i,1)*proj(spin_1_hilbert,(i+1)%N,-1)
 
    # if 1 <= i <= N-1 :
    #     H += -t/2*sigmap(spin_1_hilbert,i)*sigmam(spin_1_hilbert,(i+1)%N)*proj(spin_1_hilbert,i,0)*proj(spin_1_hilbert,(i+1)%N,0)
    #     H+= -t/2*sigmam(spin_1_hilbert,i)*sigmap(spin_1_hilbert,(i+1)%N)*proj(spin_1_hilbert,i,1)*proj(spin_1_hilbert,(i+1)%N,-1)
    # elif i ==0 : 
    #     H += l_cons*((sigmaz(spin_1_hilbert,0) -2)*(sigmaz(spin_1_hilbert,0)*(1/2)-1))
    # elif i == N-1:
    #     H+= l_cons*(sigmaz(spin_1_hilbert,N-1)*(1/2)+1)*(sigmaz(spin_1_hilbert,N-1)*(1/2)+1)
    # else : 
    #     raise(RuntimeError("invalid i"))

    if return_type == "dense":
        return H.to_dense()
    elif return_type == "sparse":
        return H.to_sparse()
    else :
        return H

def compute_energy(model, parameters, hamiltonian_sparse, hilbert, idx = None):
    psi_gs = to_array(model, parameters, hilbert)
    if idx != None :
        psi_gs = np.zeros(hilbert.all_states().shape[0])
        psi_gs[idx] = 1

    return psi_gs.conj().T@(hamiltonian_sparse@psi_gs)

def compute_state_energy(hamiltonian_sparse, hilbert, idx):
    psi_gs = np.zeros(hilbert.all_states().shape[0])
    psi_gs[idx] = 1

    return psi_gs.conj().T@(hamiltonian_sparse@psi_gs)
def diag(H):
    eig_vals, eig_vecs = eigsh(H.to_sparse(), k=2, which="SA")

    print("eigenvalues with scipy sparse:", eig_vals)

    E_gs = eig_vals[0]
    eig_vecs[:,0]