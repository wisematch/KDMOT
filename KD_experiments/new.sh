# check list:
#             kd_loss
#             head
#             pretrain
#             data_cfg
#             exp_id
#             reid_dim
#             origin_crop


# 1. CH, in 256, sl1, single head, pre det
# # train
# 30 EP, abandoned.
#cd src
#python train_KD.py kd --exp_id new_exp_1 --gpus 0,1,2,3 --batch_size 16 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/crowdhuman.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True
#cd ..

# 60 EP
#cd src
#python train_KD.py kd --exp_id new_exp_1 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/crowdhuman.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --num_epochs 60 --lr_step '50' --load_model '../models/ctdet_coco_dla_2x.pth'
#cd ..

# # test
#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_exp_1/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True
#cd ..

# 2. CH, in 256, sl1, res head, pre det
# # train
#cd src
#python train_KD.py kd --exp_id new_exp_2 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/crowdhuman.json' --kd_loss 'smooth_l1' --id_head 'res' --origin_crop True --num_epochs 60 --lr_step '50' --load_model '../models/ctdet_coco_dla_2x.pth'
#cd ..

# # test
#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_exp_2/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'res' --origin_crop True
#cd ..

# test teacher embed
# # train
# # test

# 3. CH, det only, pre det
# # train
#cd src
#python train_KD.py --task kd --exp_id new_exp_3 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/crowdhuman.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --num_epochs 60 --lr_step '50' --load_model '../models/ctdet_coco_dla_2x.pth' --id_weight 0
#cd ..

# # test
#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_exp_3/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True
#cd ..

# TODO 注意defult 和 defult_new, fastreid 和 fastreid2

# 4. CH mot17_half, in 256, sl1, single head, pre model1
# # train
#cd src
#python train_KD.py --task kd --exp_id new_exp_4 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_exp_1/model_last.pth'
#cd ..

# # test
#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_exp_4/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True
#cd ..

# 5. CH mot17_half, in 256, sl1, res head, pre model2
# # train
#cd src
#python train_KD.py --task kd --exp_id new_exp_5 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'res' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_exp_2/model_last.pth'
#cd ..

# # test
#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_exp_5/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'res' --origin_crop True
#cd ..


# 6. CH mot17_half, in 256, l1, single head, pre model1
# # train
#cd src
#python train_KD.py --task kd --exp_id new_exp_6 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_exp_1/model_last.pth'
#cd ..
# # test
cd src
python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_exp_6/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True
cd ..


# TODO 改序号
# 7. CH mot17_half, in 256, l1, res head, pre model2
# # train
# # test

# 7. CH mot17_half, in 256, l2, single head, pre model1
# # train
# # test

# 8. CH mot17_half, in 256, l2, res head, pre model2
# # train
# # test
