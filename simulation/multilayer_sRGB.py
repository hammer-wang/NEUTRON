import numpy as np
import colour
import pandas as pd
import colour
import pickle as pkl
from tmm import coh_tmm, inc_tmm
from scipy.interpolate import interp1d
from colour import SDS_ILLUMINANTS, SpectralDistribution
from colour.colorimetry import MSDS_CMFS
from colour.plotting import plot_single_colour_swatch, ColourSwatch, plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import os
import itertools
from multiprocessing import Pool
from pyDOE import lhs
import pyswarms as ps
from colour.difference import delta_E, delta_E_CIE2000
from multilayer_thin_film import load_nk
mpl.rcParams['figure.dpi'] = 200

DATABASE = '/data/hzwang/optical_data'
illuminant = SDS_ILLUMINANTS['D65']
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
pi = np.pi

D = ['TiO2', 'SiO2', 'MgF2', 'HfO2', 'Al2O3', 'ZnO', 'ZnS', 'ZnSe', 'Ta2O5', 'Fe2O3']
M = ['Ag', 'Al', 'Ni', "Ge", 'Ti', 'Zn', 'Cr', 'Cu', 'Au', 'W']
NK_DICT = load_nk(D + M)
wavelengths = np.arange(0.4, 0.701, 0.01)


def get_lab(nk_dict, thickness):
    
    R = []

    thickness = list(thickness)
    for i, lambda_vac in enumerate(wavelengths * 1e3):

        n_list = [1] + [val[i] for key, val in nk_dict.items()] + [1.5, 1]   

        inc_list = ['i', 'c', 'c', 'c', 'c', 'i', 'i', 'i']

        res = inc_tmm('s', n_list, [np.inf] + thickness + [5e4, np.inf], inc_list, 0, lambda_vac)

        res_t = inc_tmm('p', n_list, [np.inf] + thickness + [5e4, np.inf], inc_list, 0, lambda_vac)
        
        R.append((res['R'] + res_t['R']) / 2)

    data = dict(zip((1e3 * wavelengths).astype('int'), R))
    sd = SpectralDistribution(data)

    XYZ  = colour.sd_to_XYZ(sd, cmfs, illuminant)
    Lab = colour.XYZ_to_Lab(XYZ / 100)

    return Lab

pool = Pool(16)
def color_difference(x, target=None, nk_dict=None):
    thickness = np.hstack((x, 100 * np.ones((len(x), 1)))).astype(int)
    thickness = [thickness[i] for i in range(len(thickness))]
    Labs = pool.starmap(get_lab, itertools.product([nk_dict], thickness))
    Labs = np.stack(Labs)

    return np.array([delta_E_CIE2000(Lab, target) for Lab in Labs])


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--iters', default=100, type=int)
    parser.add_argument('--split', type=int, default=0)
    args = parser.parse_args()

    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    dataset = {'acc': [],
               'thickness': [],
               'target_RGB': [],
               'target_Lab': [],
               'designed_Lab': [],
               'alphas': [],
               'mats': [],
               'nk_dict': [],
               'deltaE': []}

    total_accs = 0
    rng = np.random.default_rng(args.split)

    layer_names = ['m1', 'm2', 'm3', 'm4', 'm5']
    for i in tqdm(range(args.samples)):

        # sample color target
        RGB = (rng.uniform(0, 1, size=3) * 256).astype(int)
        target = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(RGB / 255))

        # sample materials 
        if rng.binomial(1, 0.5):
            # Interpolate
            alphas = rng.uniform(0, 1, size=(5, ))
            
            # mats1:
            D1s = rng.choice(D, size=2, replace=True)
            A1s = rng.choice(M, size=2, replace=False)
            Ms1 = rng.choice(M, size=1)
            mats1 = [D1s[0], A1s[0], A1s[1], D1s[1], Ms1[0]]

            # mats2:
            D2s = rng.choice(D, size=2, replace=True)
            A2s = rng.choice(M, size=2, replace=False)
            Ms2 = rng.choice(M, size=1)
            mats2 = [D2s[0], A2s[0], A2s[1], D2s[1], Ms2[0]]

            nk_vals = [NK_DICT[mats1[i]] * alphas[i] + NK_DICT[mats2[i]] * (1 - alphas[i]) for i in range(len(layer_names))]
            nk_dict = dict(zip(layer_names, nk_vals))
            ms = mats1 + mats2
            
        else:
            alphas = np.ones(5)
            # Primitive
            Ds = rng.choice(D, size=2, replace=True)
            As = rng.choice(M, size=2, replace=False)
            Ms = rng.choice(M, size=1)
            mats = [Ds[0], As[0], As[1], Ds[1], Ms[0]]

            nk_vals = [NK_DICT[mat] for mat in mats]
            nk_dict = dict(zip(layer_names, nk_vals))

            ms = mats + mats
        
        optimizer = ps.single.GlobalBestPSO(n_particles=32, dimensions=4, options=options, bounds=([5, 5, 5, 5], [250, 15, 15, 250]), ftol=1e-4, ftol_iter=10)
        cost, pos = optimizer.optimize(color_difference, args.iters, target=target, nk_dict=nk_dict, verbose=False)

        pos = np.insert(pos, 4, 100).astype(int)
        Lab = get_lab(nk_dict, pos)
        deltaE = delta_E_CIE2000(Lab, target)

        acc = 1 if deltaE <= 2 else 0
        total_accs += acc

        print('Target RGB {}, Lab {}, designed Lab {}, deltaE {:.2f}, total accs {}'.format(RGB, np.round(target, 1), np.round(Lab, 1), deltaE, total_accs))

        dataset['acc'].append(acc)
        dataset['thickness'].append(pos)
        dataset['target_RGB'].append(RGB)
        dataset['target_Lab'].append(target)
        dataset['designed_Lab'].append(Lab)
        dataset['deltaE'].append(deltaE)
        dataset['alphas'].append(alphas)
        dataset['mats'].append(ms)
        dataset['nk_dict'].append(nk_dict)

        if (i + 1) % 100 == 0:
            
            for key, val in dataset.items():
                dataset[key] = np.array(val)

            pkl.dump(dataset, open('./multilayer_data/sRGB_additional/data_split{}_{}.pkl'.format(args.split, (i + 1) // 100), 'wb'))

            # refresh the dataset
            dataset = {'acc': [],
                       'thickness': [],
                       'target_RGB': [],
                       'target_Lab': [],
                       'designed_Lab': [],
                       'alphas': [],
                       'mats': [],
                       'nk_dict': [],
                       'deltaE': []}







            
