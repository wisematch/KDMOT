# pretrained on crowdhuman-detection

# 0. ch
cd src
python train_KD.py kd --exp_id kd_crowdhuman_dla34_res_sl1 --gpus 0,1,2,3 --batch_size 16 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/crowdhuman.json' --num_epochs 60 --lr_step '50' --kd_loss 'smooth_l1' --origin_crop True --id_head 'res'
cd ..

# 记得改load model
## 1. ch, res, sl1
#cd src
#python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34_res_sl1 --gpus 0,1,2,3 --batch_size 16 --load_model '../exp/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --origin_crop True --id_head 'res'
#cd ..
#
## 2. ch, res, mse
#cd src
#python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34_res_mse --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'mse' --origin_crop True --id_head 'res'
#cd ..

# 3.

