# input 256x128, CH FairMOT pretrained, sl1
#cd src
#python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34 --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1'
#cd ..

# input 256x128, CH FairMOT pretrained, sl1, res_head
cd src
python train_KD.py kd --exp_id kd_mot17_half_ft_ch_dla34 --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'res'
cd ..