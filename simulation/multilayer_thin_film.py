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
mpl.rcParams['figure.dpi'] = 200

DATABASE = '/data/hzwang/optical_data'
illuminant = SDS_ILLUMINANTS['D65']
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
pi = np.pi

D = ['TiO2', 'SiO2', 'MgF2', 'HfO2', 'Al2O3', 'ZnO', 'ZnS', 'ZnSe', 'Ta2O5', 'Fe2O3']
M = ['Ag', 'Al', 'Ni', "Ge", 'Ti', 'Zn', 'Cr', 'Cu', 'Au', 'W']
wavelengths = np.arange(0.4, 0.701, 0.01)

def load_nk(mats):    
    nk_dict = {}

    for mat in mats:
        nk = pd.read_csv(os.path.join(DATABASE, mat + '.csv'))
        nk.dropna(inplace=True)
        wl = nk['wl'].to_numpy()
        index = (nk['n'] + nk['k'] * 1.j).to_numpy()
        mat_nk_data = np.hstack((wl[:, np.newaxis], index[:, np.newaxis]))


        mat_nk_fn = interp1d(
                mat_nk_data[:, 0].real, mat_nk_data[:, 1], kind='quadratic')
        nk_dict[mat] = mat_nk_fn(wavelengths)
        
    return nk_dict


def get_color_(nk_dict, thickness):
    
    R = []

    thickness = list(thickness)
    for i, lambda_vac in enumerate(wavelengths * 1e3):

        n_list = [1] + [val[i] for key, val in nk_dict.items()] + [1.5, 1]   

        inc_list = ['i', 'c', 'c', 'c', 'c', 'i', 'i', 'i']

        res = inc_tmm('s', n_list, [np.inf] + thickness + [5e3, np.inf], inc_list, 0, lambda_vac)

        res_t = inc_tmm('p', n_list, [np.inf] + thickness + [5e3, np.inf], inc_list, 0, lambda_vac)
        
        R.append((res['R'] + res_t['R']) / 2)

    data = dict(zip((1e3 * wavelengths).astype('int'), R))
    sd = SpectralDistribution(data)

    XYZ  = colour.sd_to_XYZ(sd, cmfs, illuminant)
    RGB = colour.XYZ_to_sRGB(XYZ / 100)
    xyY = colour.XYZ_to_xyY(XYZ / 100)

    return xyY


def get_color(nk_dict, mats, thickness):
    
    R = []
    mats = list(mats)
    thickness = list(thickness)
    for i, lambda_vac in enumerate(wavelengths * 1e3):

        n_list = [1] + [nk_dict[mat][i] for mat in mats] + [1.5, 1]   

        inc_list = ['i', 'c', 'c', 'c', 'c', 'i', 'i', 'i']

        res = inc_tmm('s', n_list, [np.inf] + thickness + [5e3, np.inf], inc_list, 0, lambda_vac)

        res_t = inc_tmm('p', n_list, [np.inf] + thickness + [5e3, np.inf], inc_list, 0, lambda_vac)

        R.append((res['R'] + res_t['R']) / 2)

    data = dict(zip((1e3 * wavelengths).astype('int'), R))
    sd = SpectralDistribution(data)

    XYZ  = colour.sd_to_XYZ(sd, cmfs, illuminant)
    RGB = colour.XYZ_to_sRGB(XYZ / 100)
    xyY = colour.XYZ_to_xyY(XYZ / 100)
    
    fig = None

    return xyY



if __name__ == '__main__':

    radius = 0.1
    num_datasets = 0
    p = Pool(100)
    for mats in tqdm(list(itertools.product(D, M, M, D, M))):

        if mats[1] == mats[2]:
            continue
        
        designs = lhs(4, samples=100)
        t1 = designs[:, 0] * 145 + 5
        t2 = designs[:, 1] * 10 + 5
        t3 = designs[:, 2] * 10 + 5
        t4 = designs[:, 3] * 245 + 5

        nk_dict = load_nk(mats)

        xyY = p.starmap(get_color, itertools.product([nk_dict], [mats], zip(t1, t2, t3, t4, 100 * np.ones(100))))
        xyY = np.array(xyY)
        x, y, Y = xyY[:, 0], xyY[:, 1], xyY[:, 2]
        thickness = np.zeros((5, 100))
        thickness[0,:] = t1
        thickness[1,:] = t2
        thickness[2,:] = t3
        thickness[3,:] = t4
        thickness[4,:] = np.ones(100) * 100

        if np.min(np.sqrt((np.array(x) - 0.3) ** 2 + (np.array(y) - 0.6) ** 2)) < radius and np.min(np.sqrt((np.array(x) - 0.64) ** 2 + (np.array(y) - 0.33) ** 2)) < radius and np.min(np.sqrt((np.array(x) - 0.15) ** 2 + (np.array(y) - 0.06) ** 2)) < radius:
        
            num_datasets += 1
            print(num_datasets)
            pkl.dump({'Mats':mats, 'x':x, 'y':y, 'Y':Y, 'thickness':thickness}, open('./multilayer_data/{}.pkl'.format(num_datasets), 'wb'))
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plt.suptitle('{} | {} | {} | {} | {}'.format(*mats), y=1.02)
            ax[0].hist(Y, density=False)
            ax[0].set_ylabel('density')
            ax[0].set_xlabel('Y')
            ax[1].scatter(x, y, c='k', s=4, alpha=0.3)

            circle = plt.Circle((0.64, 0.33), radius, color='k', alpha=0.2)
            ax[1].scatter([0.15, 0.3, 0.64], [0.06, 0.6, 0.33], marker = '^', color='white')
            ax[1].add_patch(circle)

            circle = plt.Circle((0.3, 0.6), radius, color='k', alpha=0.2)
            ax[1].add_patch(circle)

            circle = plt.Circle((0.15, 0.06), radius, color='k', alpha=0.2)
            ax[1].add_patch(circle)
            plot_chromaticity_diagram_CIE1931(**{'axes':ax[1], 'title':None})
            fig.savefig('./multilayer_data/figs/{}.png'.format(num_datasets))


    # Just to save materials
    # radius = 0.1
    # num_datasets = 0
    # p = Pool(100)
    # for file in tqdm(glob.glob("multilayer_data/*.pkl")):
    #     data = pkl.load(open(file, 'rb'))

    #     # TODO: introduce Latin Hypercube sampling

    #     t1 = np.random.randint(5, 151, 100)
    #     t2 = np.random.randint(5, 16, 100)
    #     t3 = np.random.randint(5, 16, 100)
    #     t4 = np.random.randint(5, 251, 100)

    #     nk_dict = load_nk(data['Mats'])
    #     mats = data['Mats']
    #     data['nk_dict'] = nk_dict

    #     xyY = p.starmap(get_color, itertools.product([nk_dict], [mats], zip(t1, t2, t3, t4, 100 * np.ones(100))))
    #     xyY = np.array(xyY)
    #     x, y, Y = xyY[:, 0], xyY[:, 1], xyY[:, 2]
    #     thickness = np.zeros((100, 5))
    #     thickness[:,0] = t1
    #     thickness[:,1] = t2
    #     thickness[:,2] = t3
    #     thickness[:,3] = t4
    #     thickness[:,4] = np.ones(100) * 100

    #     # if np.min(np.sqrt((np.array(x) - 0.3) ** 2 + (np.array(y) - 0.6) ** 2)) < radius and np.min(np.sqrt((np.array(x) - 0.64) ** 2 + (np.array(y) - 0.33) ** 2)) < radius and np.min(np.sqrt((np.array(x) - 0.15) ** 2 + (np.array(y) - 0.06) ** 2)) < radius:
        
    #     num_datasets += 1
    #     print(num_datasets)
    #     pkl.dump({'Mats':mats, 'x':x, 'y':y, 'Y':y, 'thickness':thickness, 'nk_dict':nk_dict}, open('./multilayer_data/{}.pkl'.format(num_datasets), 'wb'))
            
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #     plt.suptitle('{} | {} | {} | {} | {}'.format(*mats), y=1.02)
    #     ax[0].hist(Y, density=False)
    #     ax[0].set_ylabel('density')
    #     ax[0].set_xlabel('Y')
    #     ax[1].scatter(x, y, c='k', s=4, alpha=0.3)

    #     circle = plt.Circle((0.64, 0.33), radius, color='k', alpha=0.2)
    #     ax[1].scatter([0.15, 0.3, 0.64], [0.06, 0.6, 0.33], marker = '^', color='white')
    #     ax[1].add_patch(circle)

    #     circle = plt.Circle((0.3, 0.6), radius, color='k', alpha=0.2)
    #     ax[1].add_patch(circle)

    #     circle = plt.Circle((0.15, 0.06), radius, color='k', alpha=0.2)
    #     ax[1].add_patch(circle)
    #     plot_chromaticity_diagram_CIE1931(**{'axes':ax[1], 'title':None})
    #     fig.savefig('./multilayer_data/figs/{}.png'.format(num_datasets))
