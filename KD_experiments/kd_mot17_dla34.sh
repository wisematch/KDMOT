cd src
python train_KD.py kd --exp_id kd_mot17_dla34 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17.json' --kd_loss 'smooth_l1' --origin_crop True
cd ..

cd src
python train_KD.py kd --exp_id kd_mot17_ft_ch_dla34 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17.json' --load_model '../models/crowdhuman_dla34.pth' --kd_loss 'smooth_l1' --origin_crop True
cd ..

