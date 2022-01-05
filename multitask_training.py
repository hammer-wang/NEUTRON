from re import A
import time
import numpy as np
from pytorch_lightning import callbacks
import torch
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import glob
import matplotlib as mpl
import pytorch_lightning as pl
import torch.nn.functional as F
from mdn.model import MultitaskMixtureDensityNetwork, MultitaskMixtureDensityNetworkShared
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import joblib
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import colour
from colour.difference import delta_E_CIE2000
import joblib
import json 

from re import A
import numpy as np
from pytorch_lightning import callbacks
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pickle as pkl
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import glob
import matplotlib as mpl
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import joblib
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import tqdm
from joblib import Parallel, delayed

from sklearn import metrics as sk_metrics

from tmm import coh_tmm, inc_tmm
from colour import SDS_ILLUMINANTS, SpectralDistribution
from colour.colorimetry import MSDS_CMFS
import colour
from multiprocessing import Pool
illuminant = SDS_ILLUMINANTS['D65']
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

THICKNESS_SCALER = joblib.load(os.path.join('./simulation/multilayer_data/sRGB_400K', 'thickness_scaler.joblib'))
NK_SCALER = joblib.load(os.path.join('./simulation/multilayer_data/sRGB_400K', 'nk_scaler.joblib'))
RGB_SCALER = joblib.load(os.path.join('./simulation/multilayer_data/sRGB_400K', 'rgb_scaler.joblib'))


class MultitaskDataset(Dataset):
    def __init__(self, path, split='train'):
        # files = glob.glob(os.path.join(path, '{}.pkl'))
        # num_files = len(files)
        # num_data = num_files * 100

        # self.rgb = np.zeros((num_data, 3))
        # self.thickness = np.zeros((num_data, 4))
        # self.nk = np.zeros((num_data, 310))
        # self.e = np.zeros(num_data) # DeltaE<2 label
        # self.designed_rgb = np.zeros((num_data, 3))
        
        # for i, file in tqdm.tqdm(enumerate(files)):
        #     data = pkl.load(open(file, 'rb'))
        #     start, end = i * 100, (i+1) * 100
        #     self.rgb[start:end, :] = data['target_RGB']

        #     if 'designed_RGB' not in data.keys():
        #         data['designed_RGB'] = (np.stack(list(map(self.lab_to_rgb, data['designed_Lab']))) * 255).astype(int)
        #         pkl.dump(data, open(file, 'wb'))

        #     self.designed_rgb[start:end, :] = data['designed_RGB']
        #     self.nk[start:end, :] = np.stack(list(map(self.flatten_nk, data['nk_dict'])))
        #     self.thickness[start:end, :] = data['thickness'][:, :4]
        #     self.e[start:end] = data['acc']

        data = pkl.load(open(os.path.join(path, f'{split}.pkl'), 'rb'))
        self.rgb = data['target_RGB']
        self.designed_rgb = data['designed_RGB']
        self.designed_lab = data['designed_Lab']
        self.target_lab = data['target_Lab']
        self.thickness = data['thickness']
        self.nk = np.array([self.flatten_nk(item) for item in data['nk_dict']])
        self.e = data['acc'] # DeltaE<2 label
        
        self.nk_scaler = NK_SCALER
        self.thickness_scaler = THICKNESS_SCALER
        self.rgb_scaler = RGB_SCALER
        
        # self.nk_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.nk)
        # self.rgb_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.rgb)
        # self.thickness_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.thickness)
        
        self.nk = self.nk_scaler.transform(self.nk)
        self.rgb = self.rgb_scaler.transform(self.rgb)
        self.designed_rgb = self.rgb_scaler.transform(self.designed_rgb)
        self.thickness = self.thickness_scaler.transform(self.thickness)
        
        self.rgb = torch.tensor(self.rgb).float()
        self.designed_rgb = torch.tensor(self.designed_rgb).float()
        self.nk = torch.tensor(self.nk).float()
        self.thickness = torch.tensor(self.thickness).float()[:, :4]
        self.e = torch.tensor(self.e).float()

    def __len__(self):
        return len(self.nk)

    def __getitem__(self, idx):
        return self.nk[idx], self.rgb[idx], self.designed_rgb[idx], self.thickness[idx], self.e[idx], self.target_lab[idx], self.designed_lab[idx]
    
    @staticmethod
    def flatten_nk(nk_dict):
        out = []
        for _, val in nk_dict.items():
            out.append(val.real)
            out.append(val.imag)
        return np.concatenate(out)

    @staticmethod
    def lab_to_rgb(lab):
        return colour.XYZ_to_sRGB(colour.Lab_to_XYZ(lab))

class MultitaskMDN(pl.LightningModule):

    def __init__(self, lr, hidden_dim=256, input_dim=310, n_mixtures=5, weight_decay=1e-5, alpha=1, train_ratio=1.0, dataset='400K', seed=42):
        super(MultitaskMDN, self).__init__()
        if dataset == '400K':
            self.datapath = '/data/hzwang/Projects/meta-learning-photonics-design/simulation/multilayer_data/sRGB_400K'
        elif dataset == 'primitive_100K':
            self.datapath = '/data/hzwang/Projects/meta-learning-photonics-design/simulation/multilayer_data/sRGB_primitive_100K'
        elif dataset == 'primitive_150K':
            self.datapath = '/data/hzwang/Projects/meta-learning-photonics-design/simulation/multilayer_data/sRGB_primitive_150K'
        elif dataset == 'augmented_150K':
            self.datapath = '/data/hzwang/Projects/meta-learning-photonics-design/simulation/multilayer_data/sRGB_augmented_150K'
        elif dataset == '1200K':
            self.datapath = '/data/hzwang/Projects/meta-learning-photonics-design/simulation/multilayer_data/sRGB_1200K'
        else:
            raise NotImplementedError

        self.mdn = MultitaskMixtureDensityNetwork(input_dim, 4, n_mixtures, hidden_dim, alpha)
        
        self.lr = lr
        self.wd = weight_decay
        self.alpha = alpha
        self.train_ratio = train_ratio
        self.seed = seed

    def forward(self, nk, rgb, designed_rgb, labels):
        return self.mdn(nk, rgb, designed_rgb, labels)

    def loss(self, out, thickness, labels):
        return self.mdn.loss(out, thickness, labels)

    def training_step(self, batch, batch_idx):
        nk, rgb, designed_rgb, thickness, labels, _, _ = batch
        out = list(self.forward(nk, rgb, designed_rgb, labels))
        loss = self.loss(out, thickness, labels)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            nk, rgb, designed_rgb, thickness, labels, target_lab, designed_lab_ = batch
            out = list(self.forward(nk, rgb, designed_rgb, labels))
            loss = self.loss(out, thickness, labels)

            thickness_pred = self.mdn.sample(nk, designed_rgb)[0].detach().cpu().numpy()
            thickness_pred = np.hstack((thickness_pred, np.ones((len(thickness_pred), 1))))
            thickness_pred = THICKNESS_SCALER.inverse_transform(thickness_pred).astype(int)
            thickness_pred[:, 4] = 100
            thickness_pred = thickness_pred.tolist()

            # get nk_dict
            nk_dicts = []
            nk = NK_SCALER.inverse_transform(nk.detach().cpu().numpy()).reshape(-1, 10, 31)
            for i in range(len(nk)):
                nk_dict = {"m1": nk[i, 0] + nk[i, 1] * 1j,
                        "m2": nk[i, 2] + nk[i, 3] * 1j,
                        "m3": nk[i, 4] + nk[i, 5] * 1j,
                        "m4": nk[i, 6] + nk[i, 7] * 1j,
                        "m5": nk[i, 8] + nk[i, 9] * 1j,
                        }
                nk_dicts.append(nk_dict)
            
            # target_lab = target_lab.detach().cpu().numpy().tolist()
            designed_lab_ = designed_lab_.detach().cpu().numpy().tolist()
            designed_lab = pool.starmap(get_color, zip(nk_dicts, thickness_pred))

            deltaE = np.mean([delta_E_CIE2000(lab1, lab2) for lab1, lab2 in zip(designed_lab_, designed_lab)])
            print(f'deltaE {deltaE}')


            pred = torch.sigmoid(out[-1])

            bce_loss = self.mdn.alpha * torch.nn.BCEWithLogitsLoss()(torch.squeeze(out[-1]), labels)
            labels, pred = labels.detach().cpu().numpy(), pred.detach().cpu().numpy()
            fpr, tpr, _ = sk_metrics.roc_curve(labels, pred, pos_label=1)

            #TODO: compute deltaE 

            self.log('val_loss', loss)
            self.log('val_bce_loss', bce_loss)
            self.log('val_auc', sk_metrics.auc(fpr, tpr))
            self.log('val_deltaE', deltaE)

        return loss


    def test_step(self, batch, batch_idx):
        nk, rgb, designed_rgb, thickness, labels, _ = batch
        out = list(self.forward(nk, rgb, designed_rgb, labels))
        loss = self.loss(out, thickness, labels)

        pred = torch.sigmoid(out[-1])

        bce_loss = torch.nn.BCEWithLogitsLoss()(torch.squeeze(out[-1]), labels)

        labels, pred = labels.detach().cpu().numpy(), pred.detach().cpu().numpy()
        fpr, tpr, _ = sk_metrics.roc_curve(labels, pred, pos_label=1)

        self.log('test_loss', loss)
        self.log('test_bce_loss', bce_loss)
        self.log('test_auc', sk_metrics.auc(fpr, tpr))
        return loss

    def train_dataloader(self):
        train_set = MultitaskDataset(self.datapath, 'train')
        if self.train_ratio < 1:
            num_train = int(len(train_set) * self.train_ratio)
            rng = np.random.default_rng(self.seed)
            train_indices = rng.choice(np.arange(len(train_set)), num_train, replace=False)
            train_set = Subset(train_set, train_indices)
        return DataLoader(train_set, batch_size=512, num_workers=4)

    def val_dataloader(self):
        val_set = MultitaskDataset(self.datapath, 'val')
        return DataLoader(val_set, batch_size=1000, num_workers=4, shuffle=False)

    def test_dataloader(self):
        test_set = MultitaskDataset(self.datapath, 'test')
        return DataLoader(test_set, batch_size=1000, num_workers=4, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    def compute_deltaE(self, nk, nk_dicts, rgb, lab):

        thickness_pred = self.mdn.sample(nk, rgb)[0].detach().cpu().numpy()
        thickness_pred = np.hstack((thickness_pred, np.ones((len(thickness_pred), 1))))
        thickness_pred = THICKNESS_SCALER.inverse_transform(thickness_pred).astype(int)
        thickness_pred[:, 4] = 100
        thickness_pred = thickness_pred.tolist()
        
        start_time = time.time()
        lab = lab.detach().cpu().numpy().tolist()
        lab_pred = pool.starmap(get_color, zip(nk_dicts, thickness_pred))
        end_time = time.time()

        deltaE = np.mean([delta_E_CIE2000(lab1, lab2) for lab1, lab2 in zip(lab, lab_pred)])
        print(f'Lab evaluation time {end_time - start_time:.1f} sec, deltaE {deltaE}')

        return deltaE


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--logdir', default='./logs')
    parser.add_argument('--expdir', type=str)
    parser.add_argument('--expname', default='run', type=str)
    parser.add_argument('--n_mixtures', default=32, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--train_ratio', default=1.0, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--dataset', choices=('400K', 'primitive_100K', 'primitive_150K', 'augmented_150K', '1200K'), default='400K')
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True)
    lr = args.lr
    train_ratio = args.train_ratio

    def get_color(nk_dict, thickness):
        # nk_dict, thickness = design
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
        # RGB = colour.XYZ_to_sRGB(XYZ / 100)
        Lab = colour.XYZ_to_Lab(XYZ / 100)
        # xyY = colour.XYZ_to_xyY(XYZ / 100)

        return Lab

    pool = Pool(32)

    # class DataModule(pl.LightningDataModule):
    #     datapath = '/data/hzwang/Projects/meta-learning-photonics-design/simulation/multilayer_data/sRGB_400K'

    #     train_set = MultitaskDataset(datapath, 'train')
    #     val_set = MultitaskDataset(datapath, 'val')
    #     test_set = MultitaskDataset(datapath, 'test')

    #     # For studying the sample efficiency
    #     if train_ratio < 1:
    #         num_train = int(len(train_set) * train_ratio)
    #         rng = np.random.default_rng(42)
    #         train_indices = rng.choice(np.arange(len(train_set)), num_train, replace=False)
    #         train_set = Subset(train_set, train_indices)

    #     def train_dataloader(self):
    #         return DataLoader(self.train_set, batch_size=512, num_workers=4, pin_memory=True)

    #     def val_dataloader(self):
    #         return DataLoader(self.val_set, batch_size=10000, num_workers=4, shuffle=False)

    #     def test_dataloader(self):
    #         return DataLoader(self.test_set, batch_size=10000, num_workers=4, shuffle=False)

    hparams = {'lr': args.lr,
               'n_mixtures': args.n_mixtures,
               'hidden_dim': args.hidden_dim,
               'train_ratio': args.train_ratio,
               'weight_decay': args.weight_decay,
               'alpha': args.alpha,
               'seed': args.seed}


    if not os.path.exists(os.path.join(args.logdir, args.expdir)):
        os.makedirs(os.path.join(args.logdir, args.expdir))

    input_dim = 310
    model = MultitaskMDN(lr, args.hidden_dim, input_dim, args.n_mixtures, args.weight_decay, args.alpha, args.train_ratio, args.dataset, args.seed)
    
    # data = DataModule()
    logdir = os.path.join(args.logdir, args.expdir, args.expname)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    metrics = {'loss':"val_loss"}

    with open(os.path.join(logdir, 'hparams.json'), 'w', encoding='utf-8') as f:
        json.dump(hparams, f, ensure_ascii=False, indent=4)

    trainer = pl.Trainer(max_epochs=args.epochs,
                         gpus=[args.device],
                         auto_lr_find=False,
                         default_root_dir=logdir,
                         logger=args.log,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=25,
                         callbacks=[checkpoint_callback])

    trainer.fit(model)
    # trainer.test(ckpt_path="best")
    pool.close()
