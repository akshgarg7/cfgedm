import numpy as np
import getpass
import os
import torch
from ase.data import atomic_masses
import openbabel as ob
import pybel


def compute_fingerprint(poss,numberss,num_atomss):
    fingerprint_1024 = []
    fingerprint_bits = []
    ids = len(num_atomss)
    for i in range(ids):
        pos = poss[i, :]
        numbers = numberss[i, :].squeeze()
        num_atoms = num_atomss[i]

        numbers = numbers[:num_atoms]
        pos = pos[:num_atoms]

        # minius compute mass
        m = atomic_masses[numbers]
        com = np.dot(m, pos) / m.sum()
        pos = pos - com

        # order atoms by distance to center of mass
        d = torch.sum(pos ** 2, dim=1)
        center_dists = torch.sqrt(torch.maximum(d, torch.zeros_like(d)))
        idcs_sorted = torch.argsort(center_dists)
        pos = pos[idcs_sorted]
        numbers = numbers[idcs_sorted]

        # Open Babel OBMol representation
        obmol = ob.OBMol()
        obmol.BeginModify()
        # set positions and atomic numbers of all atoms in the molecule
        for p, n in zip(pos, numbers):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(int(n))
            obatom.SetVector(*p.tolist())
        # infer bonds and bond order
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        obmol.EndModify()
        _fp = pybel.Molecule(obmol).calcfp()
        fp_bits = {*_fp.bits}
        fingerprint_bits.append(fp_bits)

        fp_32 = np.array(_fp.fp, dtype=np.uint32)
        # convert fp to 1024bit
        fp_1024 = np.array(fp_32, dtype='<u4')
        fp_1024 = torch.FloatTensor(
            np.unpackbits(fp_1024.view(np.uint8), bitorder='little'))
        fingerprint_1024.append(fp_1024)

    return fingerprint_bits,fingerprint_1024