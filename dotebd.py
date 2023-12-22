# doTEBD.py
# ---------------------------------------------------------------------
# Implementation of time evolution (real or imaginary) for MPS with 2-site unit
# cell (A-B), based on TEBD algorithm.
#
# by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 6/2019

import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigs
from ncon import ncon
from typing import Optional


def doTEBD(hamAB: np.ndarray,
           hamBA: np.ndarray,
           A: np.ndarray,
           B: np.ndarray,
           sAB: np.ndarray,
           sBA: np.ndarray,
           chi: int,
           tau: float,
           evotype: Optional[str] = 'imag',
           numiter: Optional[int] = 1000,
           midsteps: Optional[int] = 10,
           E0: Optional[float] = 0.0):
  """
  Implementation of time evolution (real or imaginary) for MPS with 2-site unit
  cell (A-B), based on TEBD algorithm.
  Args:
    hamAB: nearest neighbor Hamiltonian coupling for A-B sites.
    hamBA: nearest neighbor Hamiltonian coupling for B-A sites.
    A: MPS tensor for A-sites of lattice.
    B: MPS tensor for B-sites of lattice.
    sAB: vector of weights for A-B links.
    sBA: vector of weights for B-A links.
    chi: maximum bond dimension of MPS.
    tau: time-step of evolution.
    evotype: set real (evotype='real') or imaginary (evotype='imag') evolution.
    numiter: number of time-step iterations to take.
    midsteps: number of time-steps between re-orthogonalization of the MPS.
    E0: specify the ground energy (if known).
  Returns:
    np.ndarray: MPS tensor for A-sites;
    np.ndarray: MPS tensor for B-sites;
    np.ndarray: vector sAB of weights for A-B links.
    np.ndarray: vector sBA of weights for B-A links.
    np.ndarray: two-site reduced density matrix rhoAB for A-B sites
    np.ndarray: two-site reduced density matrix rhoAB for B-A sites
  """

  # exponentiate Hamiltonian
  d = A.shape[1]
  if evotype == "real":
    gateAB = expm(1j * tau * hamAB.reshape(d**2, d**2)).reshape(d, d, d, d)
    gateBA = expm(1j * tau * hamBA.reshape(d**2, d**2)).reshape(d, d, d, d)
  elif evotype == "imag":
    gateAB = expm(-tau * hamAB.reshape(d**2, d**2)).reshape(d, d, d, d)
    gateBA = expm(-tau * hamBA.reshape(d**2, d**2)).reshape(d, d, d, d)

  # initialize environment matrices
  sigBA = np.eye(A.shape[0]) / A.shape[0]
  muAB = np.eye(A.shape[2]) / A.shape[2]

  for k in range(numiter + 1):
    if np.mod(k, midsteps) == 0 or (k == numiter):
      """ Bring MPS to normal form """

      # contract MPS from left and right
      sigBA, sigAB = left_contract_MPS(sigBA, sBA, A, sAB, B)
      muAB, muBA = right_contract_MPS(muAB, sBA, A, sAB, B)

      # orthogonalise A-B and B-A links
      B, sBA, A = orthog_MPS(sigBA, muBA, B, sBA, A)
      A, sAB, B = orthog_MPS(sigAB, muAB, A, sAB, B)

      # normalize the MPS tensors
      A_norm = np.sqrt(ncon([np.diag(sBA**2), A, np.conj(A), np.diag(sAB**2)],
                            [[1, 3], [1, 4, 2], [3, 4, 5], [2, 5]]))
      A = A / A_norm
      B_norm = np.sqrt(ncon([np.diag(sAB**2), B, np.conj(B), np.diag(sBA**2)],
                            [[1, 3], [1, 4, 2], [3, 4, 5], [2, 5]]))
      B = B / B_norm

      """ Compute energy and display """

      # compute 2-site local reduced density matrices
      rhoAB, rhoBA = loc_density_MPS(A, sAB, B, sBA)

      # evaluate the energy
      energyAB = ncon([hamAB, rhoAB], [[1, 2, 3, 4], [1, 2, 3, 4]])
      energyBA = ncon([hamBA, rhoBA], [[1, 2, 3, 4], [1, 2, 3, 4]])
      energy = 0.5 * (energyAB + energyBA)

      chitemp = min(A.shape[0], B.shape[0])
      enDiff = energy - E0
      print('iteration: %d of %d, chi: %d, t-step: %f, energy: %f, '
            'energy error: %e' % (k, numiter, chitemp, tau, energy, enDiff))

    """ Do evolution of MPS through one time-step """
    if k < numiter:
      # apply gate to A-B link
      A, sAB, B = apply_gate_MPS(gateAB, A, sAB, B, sBA, chi)

      # apply gate to B-A link
      B, sBA, A = apply_gate_MPS(gateBA, B, sBA, A, sAB, chi)

  rhoAB, rhoBA = loc_density_MPS(A, sAB, B, sBA)
  return A, B, sAB, sBA, rhoAB, rhoBA


def left_contract_MPS(sigBA, sBA, A, sAB, B):
  """ Contract an infinite 2-site unit cell from the left for the environment
  density matrices sigBA (B-A link) and sigAB (A-B link)"""

  # initialize the starting vector
  chiBA = A.shape[0]
  if sigBA.shape[0] == chiBA:
    v0 = sigBA.reshape(np.prod(sigBA.shape))
  else:
    v0 = (np.eye(chiBA) / chiBA).reshape(chiBA**2)

  # define network for transfer operator contract
  tensors = [np.diag(sBA), np.diag(sBA), A, A.conj(), np.diag(sAB),
             np.diag(sAB), B, B.conj()]
  labels = [[1, 2], [1, 3], [2, 4], [3, 5, 6], [4, 5, 7], [6, 8], [7, 9],
            [8, 10, -1], [9, 10, -2]]

  # define function for boundary contraction and pass to eigs
  def left_iter(sigBA):
    return ncon([sigBA.reshape([chiBA, chiBA]), *tensors],
                labels).reshape([chiBA**2, 1])
  Dtemp, sigBA = eigs(LinearOperator((chiBA**2, chiBA**2), matvec=left_iter),
                      k=1, which='LM', v0=v0, tol=1e-10)

  # normalize the environment density matrix sigBA
  if np.isrealobj(A):
    sigBA = np.real(sigBA)
  sigBA = sigBA.reshape(chiBA, chiBA)
  sigBA = 0.5 * (sigBA + np.conj(sigBA.T))
  sigBA = sigBA / np.trace(sigBA)

  # compute density matric sigAB for A-B link
  sigAB = ncon([sigBA, np.diag(sBA), np.diag(sBA), A, np.conj(A)],
               [[1, 2], [1, 3], [2, 4], [3, 5, -1], [4, 5, -2]])
  sigAB = sigAB / np.trace(sigAB)

  return sigBA, sigAB


def right_contract_MPS(muAB, sBA, A, sAB, B):
  """ Contract an infinite 2-site unit cell from the right for the environment
  density matrices muAB (A-B link) and muBA (B-A link)"""

  # initialize the starting vector
  chiAB = A.shape[2]
  if muAB.shape[0] == chiAB:
    v0 = muAB.reshape(np.prod(muAB.shape))
  else:
    v0 = (np.eye(chiAB) / chiAB).reshape(chiAB**2)

  # define network for transfer operator contract
  tensors = [np.diag(sAB), np.diag(sAB), A, A.conj(), np.diag(sBA),
             np.diag(sBA), B, B.conj()]
  labels = [[1, 2], [3, 1], [5, 2], [6, 4, 3], [7, 4, 5], [8, 6], [10, 7],
            [-1, 9, 8], [-2, 9, 10]]

  # define function for boundary contraction and pass to eigs
  def right_iter(muAB):
    return ncon([muAB.reshape([chiAB, chiAB]), *tensors],
                labels).reshape([chiAB**2, 1])
  Dtemp, muAB = eigs(LinearOperator((chiAB**2, chiAB**2), matvec=right_iter),
                     k=1, which='LM', v0=v0, tol=1e-10)

  # normalize the environment density matrix muAB
  if np.isrealobj(A):
    muAB = np.real(muAB)
  muAB = muAB.reshape(chiAB, chiAB)
  muAB = 0.5 * (muAB + np.conj(muAB.T))
  muAB = muAB / np.trace(muAB)

  # compute density matrix muBA for B-A link
  muBA = ncon([muAB, np.diag(sAB), np.diag(sAB), A, A.conj()],
              [[1, 2], [3, 1], [5, 2], [-1, 4, 3], [-2, 4, 5]])
  muBA = muBA / np.trace(muBA)

  return muAB, muBA


def orthog_MPS(sigBA, muBA, B, sBA, A, dtol=1e-12):
  """ set the MPS gauge across B-A link to the canonical form """

  # diagonalize left environment matrix
  dtemp, utemp = LA.eigh(sigBA)
  chitemp = sum(dtemp > dtol)
  DL = dtemp[range(-1, -chitemp - 1, -1)]
  UL = utemp[:, range(-1, -chitemp - 1, -1)]

  # diagonalize right environment matrix
  dtemp, utemp = LA.eigh(muBA)
  chitemp = sum(dtemp > dtol)
  DR = dtemp[range(-1, -chitemp - 1, -1)]
  UR = utemp[:, range(-1, -chitemp - 1, -1)]

  # compute new weights for B-A link
  weighted_mat = (np.diag(np.sqrt(DL)) @ UL.T @ np.diag(sBA)
                  @ UR @ np.diag(np.sqrt(DR)))
  UBA, stemp, VhBA = LA.svd(weighted_mat, full_matrices=False)
  sBA = stemp / LA.norm(stemp)

  # build x,y gauge change matrices, implement gauge change on A and B
  x = np.conj(UL) @ np.diag(1 / np.sqrt(DL)) @ UBA
  y = np.conj(UR) @ np.diag(1 / np.sqrt(DR)) @ VhBA.T
  A = ncon([y, A], [[1, -1], [1, -2, -3]])
  B = ncon([B, x], [[-1, -2, 2], [2, -3]])

  return B, sBA, A


def apply_gate_MPS(gateAB, A, sAB, B, sBA, chi, stol=1e-7):
  """ apply a gate to an MPS across and a A-B link. Truncate the MPS back to
  some desired dimension chi"""

  # ensure singular values are above tolerance threshold
  sBA_trim = sBA * (sBA > stol) + stol * (sBA < stol)

  # contract gate into the MPS, then deompose composite tensor with SVD
  d = A.shape[1]
  chiBA = sBA_trim.shape[0]
  tensors = [np.diag(sBA_trim), A, np.diag(sAB), B, np.diag(sBA_trim), gateAB]
  connects = [[-1, 1], [1, 5, 2], [2, 4], [4, 6, 3], [3, -4], [-2, -3, 5, 6]]
  nshape = [d * chiBA, d * chiBA]
  utemp, stemp, vhtemp = LA.svd(ncon(tensors, connects).reshape(nshape),
                                full_matrices=False)

  # truncate to reduced dimension
  chitemp = min(chi, len(stemp))
  utemp = utemp[:, range(chitemp)].reshape(sBA_trim.shape[0], d * chitemp)
  vhtemp = vhtemp[range(chitemp), :].reshape(chitemp * d, chiBA)

  # remove environment weights to form new MPS tensors A and B
  A = (np.diag(1 / sBA_trim) @ utemp).reshape(sBA_trim.shape[0], d, chitemp)
  B = (vhtemp @ np.diag(1 / sBA_trim)).reshape(chitemp, d, chiBA)

  # new weights
  sAB = stemp[range(chitemp)] / LA.norm(stemp[range(chitemp)])

  return A, sAB, B


def loc_density_MPS(A, sAB, B, sBA):
  """ Compute the local reduced density matrices from an MPS (assumend to be
  in canonical form)."""

  # recast singular weights into a matrix
  mAB = np.diag(sAB)
  mBA = np.diag(sBA)

  # contract MPS for local reduced density matrix (A-B)
  tensors = [np.diag(sBA**2), A, A.conj(), mAB, mAB, B, B.conj(),
             np.diag(sBA**2)]
  connects = [[3, 4], [3, -3, 1], [4, -1, 2], [1, 7], [2, 8], [7, -4, 5],
              [8, -2, 6], [5, 6]]
  rhoAB = ncon(tensors, connects)

  # contract MPS for local reduced density matrix (B-A)
  tensors = [np.diag(sAB**2), B, B.conj(), mBA, mBA, A, A.conj(),
             np.diag(sAB**2)]
  connects = [[3, 4], [3, -3, 1], [4, -1, 2], [1, 7], [2, 8], [7, -4, 5],
              [8, -2, 6], [5, 6]]
  rhoBA = ncon(tensors, connects)

  return rhoAB, rhoBA
