train RankNet with tensorflow
====================

1. setup the env

the following commands install tensorflow and dependencies

```
conda create -n ranknet python=3 numpy matplotlib pandas scikit-learn scipy
conda activate ranknet
conda install -c anaconda tensorflow=2.1.0 tqdm keras

```

2. setup the benchmark directory

```
git clone git@github.com:DSC-SPIDAL/rankpredictor.git
mkdir -p test
cd test
source ../rankpredictor/bin/init_env.sh
cp ../rankpredictor/run/19.benchmark/tftrain/* .

```

3. run benchmark

```
python test_deepartf.py

```

good luck!

Test on VE
====================
1. Once the NEC TF2.3 is installed, install other requirements in requirements-ve.txt
```
pip install -r requirements-ve.txt
```

2. setup the benchmark directory

```
git clone https://github.com/DSC-SPIDAL/rankpredictor.git
mkdir -p test
cd test
source ../rankpredictor/bin/init_env.sh
cp ../rankpredictor/run/19.benchmark/tftrain/* .
```

3. run benchmark from the **test** directory

```
python testve.py

```

The model is defined here: rankpredictor/src/indycar/model/deepartfve/model_eager/lstm_ve.py

The use_veop flag can be changed here. The batsize is for changing batch size.
