cd src
python train.py mot --exp_id mot17_half_ft_ch_dla34 --gpus 0,1,2,3 --batch_size 32 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json'
cd ..