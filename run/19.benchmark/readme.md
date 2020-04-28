run benchmark
====================

1. setup the env

the following commands install the Gluonts and MxNet(cpu version)

```
conda create -n ranknet python=3 numpy matplotlib pandas scikit-learn scipy
conda activate ranknet
pip install gluonts
pip install mxnet

```

please install mxnet gpu version as the instructions on the offical website

2. setup the benchmark directory

```
git clone git@github.com:DSC-SPIDAL/rankpredictor.git
mkdir -p test
cd test
source ../rankpredictor/bin/init_env.sh
cp ../rankpredictor/run/19.benchmark/data/* .

```

3. run benchmark

```
python -m indycar.model.gluonts_models --contextlen 40 --batch_size 64 --nocarid --gpuid -1 --model deepAR --epochs 5 --input ./timediff-oracle-noip-noeid-all-all-f1min-t2-rIndy500-2018-gluonts-indy-2018.pickle --output deepAR-timediff-all-indy-f1min-t2-e1000-r1_deepar_t2

```

run on cpu by set gpuid to -1, otherwise it will run gpu.

screen output should be similar as the following:

INFO:root:Start model training" and "INFO:gluonts_models.py:Start Evaluator" are the start time point of training and inference.


```
(ranknet) [pengb@j-030 test]$ python indycar.model.gluonts_models --contextlen 40 --batch_size 32 --nocarid --gpuid -1 --model deepAR --epochs 10 --input ./timediff-oracle-noip-noeid-all-all-f1min-t2-rIndy500-2018-gluonts-indy-2018.pickle --output deepAR-timediff-all-indy-f1min-t2-e1000-r1_deepar_t2
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:gluonts_models.py:running gluonts_models.py --contextlen 40 --batch_size 32 --nocarid --gpuid -1 --model deepAR --epochs 10 --input ./timediff-oracle-noip-noeid-all-all-f1min-t2-rIndy500-2018-gluonts-indy-2018.pickle --output deepAR-timediff-all-indy-f1min-t2-e1000-r1_deepar_t2
INFO:gluonts_models.py:number of cars: [57]
INFO:gluonts_models.py:target_dim:1
INFO:gluonts_models.py:runid=-ideepAR-timediff-all-indy-f1min-t2-e1000-r1_deepar_t2-e10-mdeepAR-p2-c40-f1min-dim1-dstrstudent
INFO:root:Start model training
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
INFO:root:Epoch[0] Learning rate is 0.001
0%| | 0/100 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepARTrainingNetwork: 31056
100%|
INFO:root:Epoch[0] Elapsed time 4.656 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2.963846
INFO:root:Epoch[1] Learning rate is 0.001
100%|
...
INFO:root:Final loss: 1.9981121730804443 (occurred at epoch 8)
INFO:root:End model training
INFO:gluonts_models.py:Start to save the model to deepAR-timediff-all-indy-f1min-t2-e1000-r1_deepar_t2
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.
INFO:gluonts_models.py:End of saving the model.
INFO:gluonts_models.py:Start Evaluator
INFO:gluonts_models.py:tss len=4673, forecasts len=4673
Running evaluation: 100%|
INFO:gluonts_models.py:{
"MSE": 146.97610428600734,
"abs_error": 53546.3291456243,
"abs_target_sum": 271721.2243772391,
"abs_target_mean": 29.073531390673992,
"seasonal_error": 2.8579725079845266,
"MASE": 2.2372805521569403,
"sMAPE": 0.37048239304471203,
"MSIS": 44.47069975147491,
"QuantileLoss[0.1]": 30228.42088744048,
"Coverage[0.1]": 0.13010913759897283,
"QuantileLoss[0.5]": 53546.32911945456,
"Coverage[0.5]": 0.3286967686710892,
"QuantileLoss[0.9]": 38368.46275129839,
"Coverage[0.9]": 0.8266638133961053,
"RMSE": 12.12337017029536,
"NRMSE": 0.41698994206751955,
"ND": 0.1970634766141207,
"wQuantileLoss[0.1]": 0.11124791946864412,
"wQuantileLoss[0.5]": 0.1970634765178097,
"wQuantileLoss[0.9]": 0.14120524754455785,
"mean_wQuantileLoss": 0.14983888117700392,
"MAE_Coverage": 0.09158285184392612
}

```
