train RankNet with tensorflow
====================

1. setup the env

the following commands install the Gluonts and MxNet(cpu version)

```
conda create -n ranknet python=3 numpy matplotlib pandas scikit-learn scipy
conda activate ranknet
pip install tensorflow

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
