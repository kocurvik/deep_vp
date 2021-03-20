# This script runs training to check for viability of using non-normalized losses for training a regression models.
# Even after five epochs there is no clear trend in loss decrese.

python train_reg.py --resnet -b 32 -l mae -lr 1e-3 -e 5 -exp 30 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_orig_mae_128in_1.0s_32b_30.0.out  2> err_train_reg_VP1VP2_resnet_orig_mae_128in_1.0s_32b_30.0.out
python train_reg.py --resnet -b 32 -l mae -lr 1e-6 -e 5 -exp 60 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_orig_mae_128in_1.0s_32b_60.0.out  2> err_train_reg_VP1VP2_resnet_orig_mae_128in_1.0s_32b_60.0.out
python train_reg.py --resnet -b 32 -l mae -lr 1e-9 -e 5 -exp 90 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_orig_mae_128in_1.0s_32b_90.0.out  2> err_train_reg_VP1VP2_resnet_orig_mae_128in_1.0s_32b_90.0.out

python train_reg.py --resnet -b 32 -l mse -lr 1e-3 -e 5 -exp 30 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_orig_mse_128in_1.0s_32b_30.0.out  2> err_train_reg_VP1VP2_resnet_orig_mse_128in_1.0s_32b_30.0.out
python train_reg.py --resnet -b 32 -l mse -lr 1e-6 -e 5 -exp 60 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_orig_mse_128in_1.0s_32b_60.0.out  2> err_train_reg_VP1VP2_resnet_orig_mse_128in_1.0s_32b_60.0.out
python train_reg.py --resnet -b 32 -l mse -lr 1e-9 -e 5 -exp 90 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_orig_mse_128in_1.0s_32b_90.0.out  2> err_train_reg_VP1VP2_resnet_orig_mse_128in_1.0s_32b_90.0.out

# Training directly in the diamond space is also not viable as the mapping from the original space to the diamond space
# is not continuous

python train_reg.py --resnet --diamond -b 32 -l mae -lr 1.0 -e 5 -exp 10 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mae_128in_1.0s_32b_10.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mae_128in_1.0s_32b_10.0.out
python train_reg.py --resnet --diamond -b 32 -l mae -lr 1e-3 -e 5 -exp 30 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mae_128in_1.0s_32b_30.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mae_128in_1.0s_32b_30.0.out
python train_reg.py --resnet --diamond -b 32 -l mae -lr 1e-6 -e 5 -exp 60 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mae_128in_1.0s_32b_60.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mae_128in_1.0s_32b_60.0.out

python train_reg.py --resnet --diamond -b 32 -l mse -lr 1.0 -e 5 -exp 10 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mse_128in_1.0s_32b_10.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mse_128in_1.0s_32b_10.0.out
python train_reg.py --resnet --diamond -b 32 -l mse -lr 1e-3 -e 5 -exp 30 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mse_128in_1.0s_32b_30.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mse_128in_1.0s_32b_30.0.out
python train_reg.py --resnet --diamond -b 32 -l mse -lr 1e-6 -e 5 -exp 60 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mse_128in_1.0s_32b_60.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mse_128in_1.0s_32b_60.0.out

python train_reg.py --resnet --diamond -b 32 -l rmse -lr 1.0 -e 5 -exp 10 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_rmse_128in_1.0s_32b_10.0.out  2> err_train_reg_VP1VP2_resnet_diamond_rmse_128in_1.0s_32b_10.0.out
python train_reg.py --resnet --diamond -b 32 -l rmse -lr 1e-3 -e 5 -exp 30 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_rmse_128in_1.0s_32b_30.0.out  2> err_train_reg_VP1VP2_resnet_diamond_rmse_128in_1.0s_32b_30.0.out
python train_reg.py --resnet --diamond -b 32 -l rmse -lr 1e-6 -e 5 -exp 60 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_rmse_128in_1.0s_32b_60.0.out  2> err_train_reg_VP1VP2_resnet_diamond_rmse_128in_1.0s_32b_60.0.out

python train_reg.py --resnet --diamond -s 0.1 -b 32 -l mae -lr 1.0 -e 5 -exp 10 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mae_128in_0.1s_32b_10.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mae_128in_0.1s_32b_10.0.out
python train_reg.py --resnet --diamond -s 0.1 -b 32 -l mae -lr 1e-3 -e 5 -exp 30 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mae_128in_0.1s_32b_30.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mae_128in_0.1s_32b_30.0.out
python train_reg.py --resnet --diamond -s 0.1 -b 32 -l mae -lr 1e-6 -e 5 -exp 60 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mae_128in_0.1s_32b_60.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mae_128in_0.1s_32b_60.0.out

python train_reg.py --resnet --diamond -s 0.1 -b 32 -l mse -lr 1.0 -e 5 -exp 10 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mse_128in_0.1s_32b_10.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mse_128in_0.1s_32b_10.0.out
python train_reg.py --resnet --diamond -s 0.1 -b 32 -l mse -lr 1e-3 -e 5 -exp 30 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mse_128in_0.1s_32b_30.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mse_128in_0.1s_32b_30.0.out
python train_reg.py --resnet --diamond -s 0.1 -b 32 -l mse -lr 1e-6 -e 5 -exp 60 /home/kocurvik/BoxCars116k 1> std_train_reg_VP1VP2_resnet_diamond_mse_128in_0.1s_32b_60.0.out  2> err_train_reg_VP1VP2_resnet_diamond_mse_128in_0.1s_32b_60.0.out