# 1. CH + Mot17half, sl1, base head, 0.1, pre new_teacher_exp_1
 # train
#cd src
#python train_KD.py --task kd --exp_id standard_exp_1 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --standard_kd True --standard_kd_weight 0.1
#cd ..

 # test
#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_1/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# 2. CH + Mot17half, sl1, base head, 0.5, pre new_teacher_exp_1
 # train
#cd src
#python train_KD.py --task kd --exp_id standard_exp_2 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --standard_kd True --standard_kd_weight 0.5
#cd ..

 # test
#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_2/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..


# ---------




# 3. CH + Mot17half, sl1, base head, auto, pre new_teacher_exp_1
# train
#cd src
#python3 train_KD.py --task kd --exp_id standard_exp_3 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth' --standard_kd True --num_epochs 60 --lr_step '50'
#cd ..

# test
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_last.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_30.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_59.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_58.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_57.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_56.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_55.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_54.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_53.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_52.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_51.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_50.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..
#
#cd src
#python3 track_half.py --task kd --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_40.pth' --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..



# 4. person search, pre new_teacher_exp_1, auto, ep 60, new teacher
#cd src
#python3 train_KD.py --task kd --exp_id standard_ps_1 --gpus 0,1,3 --batch_size 12 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/home/helingxiao3/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth' --standard_kd True --num_epochs 60 --lr_step '50'
#cd ..

# resume
#cd src
#python3 train_KD.py --task kd --exp_id standard_ps_1 --gpus 0,1,3 --batch_size 12 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --resume --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/home/helingxiao3/wei.zhang/pretrained/teachers/model_final.pth' --standard_kd True --num_epochs 60 --lr_step '50'
#cd ..

# test
#cd src
#python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_exp_3/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
#cd ..

# 5. ablation for standard KD, teacher + label
# train
#cd src
#python train_KD.py --task kd --exp_id standard_mot_1 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth' --standard_kd True --num_epochs 30 --lr_step '20'
#cd ..

# test
cd src
python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_mot_1/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
cd ..

# 6. ablation for standard KD, mot teacher + label
# train
#cd src
#python train_KD.py --task kd --exp_id standard_mot_2 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/mot17_half.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '/export/wei.zhang/PycharmProjects/FairMOT/exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/mot17_teacher.pth' --standard_kd True --num_epochs 30 --lr_step '20'
#cd ..

# test
cd src
python track_half.py --task kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/standard_mot_2/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'base' --origin_crop True --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/model_final.pth'
cd ..

