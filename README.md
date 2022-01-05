# NEUTRON: Neural Particle Swarm Optimization for Material-Aware Inverse Design of Structural Color

This is the official repository accompanying the paper __NEUTRON: Neural Particle Swarm Optimization for Material-Aware Inverse Design of Structural Color__. [link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3992098)

# Installation
1. clone the repository to your local machine: `git clone https://github.com/hammer-wang/NEUTRON.git`
2. create a conda envionrment using the provided env configuration file: `conda env create -f environment.yml`

# Download the dataset
Download the train/val/test splits from [link](https://data.mendeley.com/datasets/54d84d88p8/draft?a=1d40c31c-0439-4078-99f8-5c666b869ec9). Store the files to `./simulation/multilayer_data/sRGB_400K/`.

# Training model
1. activate the conda environment: `conda activate meta-learning`
2. run the bash script `bash ./exp_script/run_best.sh`  
The model checkpoint will be automatically save the to the folde `./log/` for downstream evaluations.

# Cr design experiment
Please refer to the provided `paper_figures.ipynb` notebook.

# Photo reconstruction
Please refer to the bash script `./exp_scripts/reconstruct_imgs.sh`. 

____________
If you find this repository useful for your research, please consider citing as:  

```
@article{wang3992098neutron,
  title={Neutron: Neural Particle Swarm Optimization for Material-Aware Inverse Design of Structural Color},
  author={Wang, Haozhu and Guo, L Jay},
  journal={Available at SSRN 3992098}
}
```
