# Traffic Camera Calibration via Vehicle Vanishing Point Detection

This repository contains the code for the paper "Traffic Camera Calibration via Vehicle Vanishing Point Detection" (submitted to ICANN 2021, arxiv: TBA)

### Results

The results from the paper can be downloaded from [Google Drive](https://drive.google.com/file/d/1JfG1kZQf82I5y9z5kAom-zlsh8Hw7qvG/view?usp=sharing). The results need to placed in the relevant folders of the BrnoCompSpeed and BrnoCarPark dataset. The results zip also contains the files for methods by other authors which we used for comparison in our paper.

You can run the evaluation using the following:
```
python eval/eval_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark
```

### Model preview

You can download the pretrained model from from [Google Drive](https://drive.google.com/file/d/1Ppz96cIEol_UqgF2mQpoTxKkGwVvQ38P/view?usp=sharing) and extract the contents into the repository directory.

You can run the preview using the preview_heatmap.py:

```
python preview_heatmap.py -b 32 -ps 25.0 -cd 10 -exp 1 -r 75 /path/to/video/or/folder/containing/images
```

### Training

To train the main model from the paper run:

```
python train_heatmap.py -b 32 -ps 25.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_heatmap.py -b 32 -ps 25.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 75 /path/to/BoxCars116k
```

You will need to download the BoxCars116k dataset.

### Model evaluation

To evaluate the model first run the object detectors on BrnoCompSpeed and BrnoCarPark datasets:

```
python eval/detect_bcp.py /path/to/BrnoCarPark
python eval/detect_bcs.py --skip 10 /path/to/2016-ITS-BrnoCompSpeed
```

Then extract the vanishing points:

```
python eval/extract_vp_bcs_heatmap.py --skip 10 -b 32 -ps 25.0 -cd 10 -exp 1 -r 75 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_heatmap.py -b 32 -ps 25.0 -cd 10 -exp 1 -r 75 /path/to/BrnoCarPark
```

Finally, extract the camera calibration file and run the evaluation script:

```
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_64out_4s_2n_32b_256c_1_r75
python eval/eval_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark
```

### Ablation experiments

To run all of the ablation experiments from Table 3 in the paper run `scripts/run_experiments.sh`. Note that you will need to run `scripts/prepare_object_detection.sh` if you haven't dones so before.