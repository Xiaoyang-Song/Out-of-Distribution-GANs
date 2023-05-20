
from ood_gan import *
from models.dc_gan_model import *
from dataset import *
from config import *
from models.model import *
from eval import *
import argparse
import time
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Training configuration file')
parser.add_argument('--n_ood', help="Number of OoD samples", type=int)
args = parser.parse_args()
assert args.config is not None, 'Please specify the config .yml file to proceed.'
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

#---------- Argument Processing  ----------#
n_ood = args.n_ood
dataset, is_within, ind, ood = config['dataset'].values()
sample_dir = config['path']['sample_dir']
regime, observed_cls = config['experiment'].values()

sample_dir += f"{dataset}/"
os.makedirs(sample_dir, exist_ok=True)

#---------- Sampling  ----------#
# Note that the batch size here is just dummy arguments
dset = DSET(dataset, is_within, 256, 256, ind, ood)


if regime == 'Balanced':
    ood_img_batch, ood_img_label = dset.ood_sample(n_ood, regime)
elif regime == 'Imbalanced':
    ood_img_batch, ood_img_label = dset.ood_sample(
        n_ood, regime, observed_cls)

fname = sample_dir + f"OOD-{regime}-{n_ood}.pt"
torch.save((ood_img_batch, ood_img_label), fname)
print("Sampling is successful!")
print(f"Labels: {ood_img_label}")




