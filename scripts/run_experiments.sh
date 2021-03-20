# This script trains and extracts calibration files for the 10 models presented in Table 3.

# Model 1 is the main model of the paper
python train_heatmap.py -b 32 -ps 25.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_heatmap.py -b 32 -ps 25.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 75 /path/to/BoxCars116k

python eval/extract_vp_bcs_heatmap.py --skip 10 -b 32 -ps 25.0 -cd 10 -exp 1 -r 75 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_heatmap.py -b 32 -ps 25.0 -cd 10 -exp 1 -r 75 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_64out_4s_2n_32b_256c_1_r75

# Model 2 with less augmentation
python train_heatmap.py -b 32 -ps 10.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_heatmap.py -b 32 -ps 10.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 75 /path/to/BoxCars116k

python eval/extract_vp_bcs_heatmap.py --skip 10 -b 32 -ps 10.0 -cd 10 -exp 1 -r 75 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_heatmap.py -b 32 -ps 10.0 -cd 10 -exp 1 -r 75 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2b_pd_aug_10.0ps_10cd_128in_64out_4s_2n_32b_256c_1_r75

# Model 3 with no augmentation
python train_heatmap.py -b 32 -ps 10.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_heatmap.py -b 32 -ps 10.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 75 /path/to/BoxCars116k

python eval/extract_vp_bcs_heatmap.py --skip 10 -b 32 -ps 0.0 -cd 0 -exp 1 -r 75 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_heatmap.py -b 32 -ps 0.0 -cd 0 -exp 1 -r 75 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2b_pd_noaug_128in_64out_4s_2n_32b_256c_1_r75

# Model 4 with only one heatmap per VP at scale 0.03
python train_heatmap.py -b 32 -s 0.03 -ps 25.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_heatmap.py -b 32 -s 0.03 -ps 25.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 75 /path/to/BoxCars116k

python eval/extract_vp_bcs_heatmap.py --skip 10 -s 0.03 -b 32 -ps 25.0 -cd 10 -exp 1 -r 75 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_heatmap.py -b 32 -s 0.03 -ps 25.0 -cd 10 -exp 1 -r 75 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_64out_0.03s_2n_32b_256c_1_r75

# Model 5 with only one heatmap per VP at scale 0.1
python train_heatmap.py -b 32 -s 0.1 -ps 25.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_heatmap.py -b 32 -s 0.1 -ps 25.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 75 /path/to/BoxCars116k

python eval/extract_vp_bcs_heatmap.py --skip 10 -s 0.1 -b 32 -ps 25.0 -cd 10 -exp 1 -r 75 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_heatmap.py -b 32 -s 0.1 -ps 25.0 -cd 10 -exp 1 -r 75 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_64out_0.1s_2n_32b_256c_1_r75

# Regression models ouput the VPs directly
# Model 6 - resnet model with normalized L1 loss
python train_reg.py -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 80 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -lr 0.00001 -expr 1 -r 80 -exp 2 -e 100 /path/to/BoxCars116k

python eval/extract_vp_bcs_reg.py --skip 10 -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -exp 2 -r 100 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_reg.py -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -exp 2 -r 100 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2_resnet_orig_normalized_aug_25.0ps_10cd_128in_1.0s_32b_2_r100

# Model 7 - resnet model with normalized L1 loss
python train_reg.py -b 32 --resnet -l n1.0 -ps 10.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l n1.0 -ps 10.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 80 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l n1.0 -ps 10.0 -cd 10 -lr 0.00001 -expr 1 -r 80 -exp 2 -e 100 /path/to/BoxCars116k

python eval/extract_vp_bcs_reg.py --skip 10 -b 32 --resnet -l n1.0 -ps 10.0 -cd 10 -exp 2 -r 100 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_reg.py -b 32 --resnet -l n1.0 -ps 10.0 -cd 10 -exp 2 -r 100 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2_resnet_orig_normalized_aug_10.0ps_10cd_128in_1.0s_32b_2_r100

# Model 8 - resnet model with normalized L2 loss
python train_reg.py -b 32 --resnet -l s1.0 -ps 25.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l s1.0 -ps 25.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 80 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l s1.0 -ps 25.0 -cd 10 -lr 0.00001 -expr 1 -r 80 -exp 2 -e 100 /path/to/BoxCars116k

python eval/extract_vp_bcs_reg.py --skip 10 -b 32 --resnet -l s1.0 -ps 25.0 -cd 10 -exp 2 -r 100 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_reg.py -b 32 --resnet -l s1.0 -ps 25.0 -cd 10 -exp 2 -r 100 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2_resnet_orig_sqr_aug_25.0ps_10cd_128in_1.0s_32b_2_r100

# Model 9 - resnet model with normalized L2 loss
python train_reg.py -b 32 --resnet -l s1.0 -ps 10.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l s1.0 -ps 10.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 80 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l s1.0 -ps 10.0 -cd 10 -lr 0.00001 -expr 1 -r 80 -exp 2 -e 100 /path/to/BoxCars116k

python eval/extract_vp_bcs_reg.py --skip 10 -b 32 --resnet -l s1.0 -ps 10.0 -cd 10 -exp 2 -r 100 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_reg.py -b 32 --resnet -l s1.0 -ps 10.0 -cd 10 -exp 2 -r 100 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2_resnet_orig_sqr_aug_10.0ps_10cd_128in_1.0s_32b_2_r100

# Model 10 - hourglass model with normalized L1 loss
python train_reg.py -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -lr 0.001 -exp 0 -e 60 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 80 /path/to/BoxCars116k
python train_reg.py -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -lr 0.00001 -expr 1 -r 80 -exp 2 -e 100 /path/to/BoxCars116k

python eval/extract_vp_bcs_reg.py --skip 10 -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -exp 2 -r 100 /path/to/2016-ITS-BrnoCompSpeed
python eval/extract_vp_bcp_reg.py -b 32 --resnet -l n1.0 -ps 25.0 -cd 10 -exp 2 -r 100 /path/to/BrnoCarPark
python eval/extract_calib.py /path/to/2016-ITS-BrnoCompSpeed /path/to/BrnoCarPark VPout_VP1VP2_reg_orig_normalized_aug_25.0ps_10cd_b_128in_64out_1.0s_64f_2n_32b_256c_1_r75


