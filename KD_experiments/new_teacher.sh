# check list:
#             kd_loss
#             head
#             pretrain
#             data_cfg
#             exp_id
#             reid_dim
#             origin_crop



# 1. CH, in 256, sl1, base head, pre det
# # train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_1 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/crowdhuman.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --num_epochs 60 --lr_step '50' --load_model '../models/ctdet_coco_dla_2x.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

 # test
#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..


# 2. CH + Mot17half, sl1, base head, pre new_teacher_exp_1
# # train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_2 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

 # test
#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_2/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# 3. CH + Mot17full, sl1, base head, pre new_teacher_exp_1
# # train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_3 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# # test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_3/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot17 True
#cd ..
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_3/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot17 True
#cd ..


# 4. CH, in 256, sl1, res head, pre det
# Not implement, out of CUDA memory


# 5. fair Mix, mot17full, in 256, sl1, res head, pre mix
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_5 --gpus 0,1,2,3 --batch_size 16  --data_cfg '../src/lib/cfg/mot17.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/pretrained/pretrained/fairmot/mix_mot17_half_dla34.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# # test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_5/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot17 True
#cd ..
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_5/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot17 True
#cd ..

# 6. fair Mix, mot17full, 60, in 256, sl1, res head, pre mix
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_6 --gpus 0,1,2,3 --batch_size 16  --data_cfg '../src/lib/cfg/mot17.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --num_epochs 60 --lr_step '50'
#cd ..

# test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_6/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot17 True
#cd ..
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_6/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot17 True
#cd ..

# 7. CH + Mot16, sl1, base head, pre new_teacher_exp_1
## # train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_7 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot16.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_7/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot16 True
#cd ..
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_7/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot16 True
#cd ..


# 8. CH + Mot16full, fair mot, b16
# train
#cd src
#python train.py --task mot --exp_id new_teacher_exp_8 --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot16.json' --reid_dim 128
#cd ..

# test
#cd src
#python track.py --task mot --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/mot/new_teacher_exp_8/model_last.pth --conf_thres 0.4 --id_head 'base' --val_mot16 True --reid_dim 128
#cd ..
#
#cd src
#python track.py --task mot --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/mot/new_teacher_exp_8/model_last.pth --conf_thres 0.4 --id_head 'base' --test_mot16 True --reid_dim 128
#cd ..



# 9. Mix, in 256, sl1, base head, pre det
# train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_9 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/data.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../models/ctdet_coco_dla_2x.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_9/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot17 True
#cd ..
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_9/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot17 True
#cd ..

# 10. Mix+Mot17, in 256, sl1, base head, pre mix

# train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_10 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_9/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

 #test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_9/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot17 True
#cd ..
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_9/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot17 True
#cd ..



# 11. Mix+Mot16, in 256, sl1, base head, pre mix

# train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_11 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot16.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_9/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_11/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot16 True
#cd ..
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_11/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot16 True
#cd ..

# 12. Mix+Mot17, in 256, sl1, base head, pre mix, id weight 1000

# train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_12 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_9/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --id_weight 1000
#cd ..

# 13. retrain Mix, in 256, sl1, base head, pre det, id_weight 100
# train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_mix --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/data.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../models/ctdet_coco_dla_2x.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --id_weight 100
#cd ..

# test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_11/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot17 True
#cd ..
#
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_11/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot17 True
#cd ..



# 15. retrain Mix, in 256, sl1, base head, pre CH, MOT17
## train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_mixCH --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/data.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_mixCH/model_30.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot17 True
#cd ..
##
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_mixCH/model_30.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot17 True
#cd ..

# 16. retrain Mix, in 256, sl1, base head, pre CH, MOT16
# train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_mixCH_16 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot16.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_mixCH/model_30.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..



# 17. CH + Mot17full, fair mot, b16
# train
#cd src
#python train.py --task mot --exp_id new_teacher_exp_fair_17 --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17.json' --reid_dim 128
#cd ..

# test
#cd src
#python track.py --task mot --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/mot/new_teacher_exp_fair_17/model_last.pth --conf_thres 0.4 --id_head 'base' --val_mot17 True --reid_dim 128
#cd ..
#
#cd src
#python track.py --task mot --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/mot/new_teacher_exp_fair_17/model_last.pth --conf_thres 0.4 --id_head 'base' --test_mot17 True --reid_dim 128
#cd ..

# test
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_mixCH_16/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --val_mot16 True
#cd ..
##
#cd src
#python track.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_mixCH_16/model_last.pth --conf_thres 0.4 --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --test_mot16 True
#cd ..


# 18. MOT17 only, in 256, sl1, base head, pre DET
## train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_exp_mot17only --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../models/ctdet_coco_dla_2x.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_mot17only/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# 19. MOT17 only, in 256, sl1, base head, pre exp1

# # train
#cd src
#python train_KD.py --task kd --exp_id new_teacher_mot17_teacher --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/mot17_teacher.pth'
#cd ..

 # test
#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_mot17_teacher/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/mot17_teacher.pth'
#cd ..

# 20. CH + Mot17half, fair mot, b16
# train
#cd src
#python train.py --task mot --exp_id new_teacher_mot17_fairmot --gpus 0,1,2,3 --batch_size 16 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --reid_dim 128
#cd ..

cd src
python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/mot/new_teacher_mot17_fairmot/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/mot17_teacher.pth'
cd ..


# -------------------------------------------
# test ablation
# only iou
#cd src
#python track_half.py --task kd --ablation_only_iou True --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_2/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# only embed
#cd src
#python track_half.py --task kd --ablation_only_embed True --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_2/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# FairMOT only iou
#cd src
#python track_half.py --task kd --ablation_only_iou True --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_2/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..


# 9H
