cd src
python train_KD.py kd --exp_id kd_crowdhuman_dla34 --gpus 0,1,2,3 --batch_size 16 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json' --kd_loss 'smooth_l1' --origin_crop True --resume
cd ..