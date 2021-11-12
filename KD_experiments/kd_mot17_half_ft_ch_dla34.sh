# pretrained on crowdhuman-detection

# exp 1, kd loss l1
#cd src
#python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34 --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json'
#cd ..

# exp 2, kd loss sl1
#cd src
#python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34_sl1 --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1'
#cd ..

# exp 3, kd loss l2
#cd src
#python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34_mse --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'mse'
#cd ..

# exp 4, res head + sl1
#cd src
#python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34_rh --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1'
#cd ..

# exp 5, sl1, origin crop
#cd src
#python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34_oc --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --origin_crop True
#cd ..

cd src
python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34_oc_res --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --origin_crop True --id_head 'res'
cd ..

cd src
python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34_oc_res_mse --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'mse' --origin_crop True --id_head 'res'
cd ..

cd src
python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34_oc_mse --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'mse' --origin_crop True
cd ..