# Neural-architecture-search

## Introduction
This is a tutorial for Neural Architecture Search, the training model is **Darts** and the data set is **CIFAR-10**.
## Environment
* Windows10
* CUDA 10.2
* cuDNN v8.0.2
* GPU: NVIDIA GeForce GTX 1660 Ti

## Installation
1. **Create a conda environment**
```
conda create -n neuralsearch python=3.8 -y
conda activate neuralsearch
```
2. **Install PyTorch 1.6.0+cu101**
```
pip install torch==1.6.0 torchvision==0.7.0
```
## Data
CIFAR-10 can be automatically downloaded by torchvision when searching the architecture
## Architecture search
1. **Modify```train_search.py```**  
Save the path and remove the timestamp
```
args.save = 'search-{}'.format(args.save)
```
Remove the index behind ```data``` (torch version < 1.0.0 is not applicable)
```
 objs.update(loss.data, n)
 top1.update(prec1.data, n)
 top5.update(prec5.data, n)
```
2. **To carry out architecture search using 2nd-order approximation, run**
```
cd cnn
python train_search.py --unrolled
```
3. **Modify '''utils.py'''**
```
def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  #print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    #os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
```
4. **Do step 2 again**
## Architecture evaluation
1. **Copy the searched network model from ```search-EXP\log.txt```:**  
Example
```
genotype = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('skip_connect', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
```
2.**Paste the model to '''genotypes.py'''**
```
DARTS_V3 = Genotype(!!! model !!!)
DARTS = DARTS_V3
```
3. **Modify```train.py```**  
Save the path and remove the timestamp
```
args.save = 'search-{}'.format(args.save)
```
Remove the index behind ```data``` (torch version < 1.0.0 is not applicable)
```
 objs.update(loss.data, n)
 top1.update(prec1.data, n)
 top5.update(prec5.data, n)
```
4. **To evaluate our best cells by training from scratch, run**
```
cd cnn && python train.py --auxiliary --cutout
```
## Visualization
**Package graphviz is required to visualize the learned cells**
```
python visualize.py DARTS
```
