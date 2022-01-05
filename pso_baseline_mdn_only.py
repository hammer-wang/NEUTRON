import glob
import itertools
import json
import multiprocessing
import os
import pickle as pkl
import time

import colour
import numpy as np
import torch
from colour import SDS_ILLUMINANTS, SpectralDistribution
from colour.colorimetry import MSDS_CMFS
from PIL import Image
from pyswarms.utils.functions import single_obj as fx
from tmm import inc_tmm
from tqdm import tqdm

from multitask_training import MultitaskDataset, MultitaskMDN
from simulation.multilayer_thin_film import load_nk

illuminant = SDS_ILLUMINANTS['D65']
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
pi = np.pi

def get_color(nk_dict, thickness):
    
    R = []
    wavelengths = np.arange(0.4, 0.701, 0.01)

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
    Lab = colour.XYZ_to_Lab(XYZ / 100)
    xyY = colour.XYZ_to_xyY(XYZ / 100)

    return XYZ, xyY, RGB, Lab

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--n_mixtures', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--pop_size', default=32, type=int)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--painting_path', type=str)
    args = parser.parse_args()

    painting = args.painting_path

    D = ['TiO2', 'SiO2', 'MgF2', 'HfO2', 'Al2O3', 'ZnO', 'ZnS', 'ZnSe', 'Ta2O5', 'Fe2O3']
    M = ['Ag', 'Al', 'Ni', "Ge", 'Ti', 'Zn', 'Cr', 'Cu', 'Au', 'W']
    NK_DICT = load_nk(D + M)

    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    model_path = args.model_path
    painting_name = args.painting_path.split('/')[2].strip('.jpg')
    print('Reconstructing {}'.format(painting_name))

    device = args.device
    model_path = glob.glob('./logs/1202_nmixtures_400K/nmix51200_400K/lightning_logs/version_0/checkpoints/*.ckpt')[0]
    f = open('./logs/1202_nmixtures_400K/nmix51200_400K/hparams.json')
    data = json.load(f)
    f.close()
    model = MultitaskMDN.load_from_checkpoint(model_path, lr=data['lr'], hidden_dim=data['hidden_dim'], n_mixtures=data['n_mixtures'], train_ratio=data['train_ratio'], alpha=data['alpha'], seed=data['seed'], weight_decay=data['weight_decay'])
    model = model.to(device)

    datapath = './simulation/multilayer_data/sRGB_400K'
    dataset = MultitaskDataset(datapath)
    quantize = True
    
    pool = multiprocessing.Pool(args.pop_size)

    image = Image.open(painting)
    image.thumbnail((512, 512), Image.ANTIALIAS)
    RGB = np.array(image).reshape(-1, 3)

    if quantize:
        step_size = 10
        RGB_quantize = (RGB / step_size).astype(int) * step_size
        RGB = np.unique(RGB_quantize, axis=0)

    # Compute Lab target
    Labs_target = [colour.XYZ_to_Lab(colour.sRGB_to_XYZ(item)) for item in tqdm(RGB / 255)]

    # get_nk_input
    RGB_input = torch.tensor(dataset.rgb_scaler.transform(RGB)).float().to(device)

    mats_list = list(itertools.product(D, M, M, D, M))

    ratio_file = f'./paper_figures/reconstruction/{painting_name}_ratios.pkl'
    if os.path.exists(ratio_file):
        ratios = pkl.load(open(f'./paper_figures/reconstruction/{painting_name}_ratios.pkl', 'rb'))

    else:
        ratios = []
        dummy_label = torch.zeros(len(RGB_input)).to(device)
        with torch.no_grad():
            for mats in tqdm(mats_list):
                nk_dict = dict(zip(['m1', 'm2', 'm3', 'm4', 'm5'], [NK_DICT[mat] for mat in mats]))
                nk = dataset.nk_scaler.transform(np.tile(dataset.flatten_nk(nk_dict), (len(RGB), 1)))
                nk_input = torch.tensor(nk).float().to(device)
                out = model.forward(nk_input, RGB_input, RGB_input, dummy_label)
                ratios.append(torch.sigmoid(out[3]).mean().item())
        pkl.dump(ratios, open(f'./paper_figures/reconstruction/{painting_name}_ratios.pkl', 'wb'))

    best_idx = np.argmax(ratios)
    mats = mats_list[best_idx]
    nk_dict = dict(zip(['m1', 'm2', 'm3', 'm4', 'm5'], [NK_DICT[mat] for mat in mats]))
    nk = dataset.nk_scaler.transform(np.tile(dataset.flatten_nk(nk_dict), (len(RGB), 1)))
    nk_input = torch.tensor(nk).float().to(device)
    print('Best material {}, avg prob {}'.format(mats, ratios[best_idx]))

    start_time = time.time()
    design_list = []
    with torch.no_grad():
        design, pred = model.mdn.sample(nk_input, RGB_input)
        design = design.detach().cpu().numpy()
        design = np.hstack((design, np.zeros((len(design), 1))))
        design = dataset.thickness_scaler.inverse_transform(design)
        design_list.append(design)
    design_list = np.array(design_list).squeeze()

    # map location to the unique row id
    shape = np.array(image).shape
    RGB_quantize = RGB_quantize.reshape(shape)
    loc_to_id = {}
    for i in range(shape[0]):
        for j in range(shape[1]):
            loc_to_id[(i, j)] = np.where(np.abs((RGB - RGB_quantize[i][j])).sum(axis=1) == 0)[0][0]

    res_dict = {}
    all_targets = []
    all_thickness = []
    nk_dicts = []

    end_time = time.time()
    all_thickness = design_list
    all_thickness = np.array(all_thickness).astype(int)

    all_thickness = design_list.tolist()
    res = pool.starmap(get_color, itertools.product([nk_dict], all_thickness))
    RGBs_designed = [item[2] for item in res]
    Labs_designed = [item[3] for item in res]

    deltaE = [colour.difference.delta_E_CIE2000(lab1, lab2) for lab1, lab2 in zip(Labs_designed, Labs_target)]

    print(np.mean(deltaE))

    reconstructed = np.zeros_like(np.array(image)).astype(float)
    shape = reconstructed.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            reconstructed[i, j] = RGBs_designed[loc_to_id[(i, j)]]
    
    image = Image.fromarray(np.uint8(reconstructed * 255))
    image.save(f'./paper_figures/reconstruction/{painting_name}_recon_mdn_only_time{end_time - start_time:.2f}sec_deltaE{np.mean(deltaE):.2f}.jpg')
      
    pool.close()
