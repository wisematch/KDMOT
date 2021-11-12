# check list:
#             kd_loss
#             head
#             pretrain
#             data_cfg
#             exp_id
#             reid_dim
#             origin_crop


# 1. prw, det only, pre det
# # train
#cd src
#python train_KD.py --exp_id ps_exp_1 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_exp_1/model_last.pth'
#cd ..

# 2. prw, l2, pre det
# # train
#cd src
#python3 train_KD.py --exp_id ps_exp_2 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'mse' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_exp_1/model_last.pth' --model_t_path "/home/helingxiao3/fast-reid1/projects/bjzProject/logs/bjz/test_ram/model_final.pth"
#cd ..

# 3. pre det, id weight * 100
# # train
#cd src
#python3 train_KD.py --exp_id ps_exp_3 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_exp_1/model_last.pth' --model_t_path "/home/helingxiao3/fast-reid1/projects/bjzProject/logs/bjz/test_ram/model_final.pth" --id_weight 100
#cd ..

# 4. pre det, id weight * 10000
# # train 应该是ps_exp_4, 错写成3
#cd src
#python3 train_KD.py --exp_id ps_exp_3 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_exp_1/model_last.pth' --model_t_path "/home/helingxiao3/fast-reid1/projects/bjzProject/logs/bjz/test_ram/model_final.pth" --id_weight 10000
#cd ..

# 5. pre det, id weight * 1000
#cd src
#python3 train_KD.py --exp_id ps_exp_5 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_exp_1/model_last.pth' --model_t_path "/home/helingxiao3/fast-reid1/projects/bjzProject/logs/bjz/test_ram/model_final.pth" --id_weight 1000
#cd ..

# 6. pre det, id weight * 100
#cd src
#python3 train_KD.py --exp_id ps_exp_6 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_exp_1/model_last.pth' --model_t_path "/home/helingxiao3/fast-reid1/projects/bjzProject/logs/bjz/test_ram/model_final.pth" --id_weight 100
#cd ..

# 7. pre det, id weight * 500
#cd src
#python3 train_KD.py --exp_id ps_exp_7 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_exp_1/model_last.pth' --model_t_path "/home/helingxiao3/fast-reid1/projects/bjzProject/logs/bjz/test_ram/model_final.pth" --id_weight 500
#cd ..

# 8. pre det, id weight * 100, ep 60
#cd src
#python3 train_KD.py --exp_id ps_exp_8 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_exp_1/model_last.pth' --model_t_path "/home/helingxiao3/fast-reid1/projects/bjzProject/logs/bjz/test_ram/model_final.pth" --id_weight 100 --num_epochs 60 --lr_step '50'
#cd ..

# 9. pre det, id weight * 1000, ep 120
#cd src
#python3 train_KD.py --exp_id ps_exp_9 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_exp_1/model_last.pth' --model_t_path "/home/helingxiao3/fast-reid1/projects/bjzProject/logs/bjz/test_ram/model_final.pth" --id_weight 1000 --num_epochs 120 --lr_step '80, 100'
#cd ..

# 10. pre det, id weight * 1000, ep 150, new teacher
#cd src
#python3 train_KD.py --exp_id ps_exp_10 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/alldomain_r50ibn_0001/model_best.pth' --id_weight 1000 --num_epochs 150 --lr_step '90, 120, 140'
#cd ..

# 11. pre det, id weight * 1000, ep 150
#cd src
#python3 train_KD.py --exp_id ps_exp_9 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --model_t_path "/home/helingxiao3/fast-reid1/projects/bjzProject/logs/bjz/test_ram/model_final.pth" --id_weight 1000 --num_epochs 150 --lr_step '80, 100, 120, 140' --resume
#cd ..

# 12. pre det, id weight * 1000, ep 30, new teacher
#cd src
#python3 train_KD.py --exp_id ps_exp_12 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/alldomain_r50ibn_0001/model_best.pth' --id_weight 1000
#cd ..

# 13. pre mix, id weight * 1000, ep 30, new teacher
#cd src
#python3 train_KD.py --exp_id ps_exp_13 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_teacher_exp_9/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/alldomain_r50ibn_0001/model_best.pth' --id_weight 1000
#cd ..

# 14. pre mix, id weight * 1000, ep 120, new teacher
#cd src
#python3 train_KD.py --exp_id ps_exp_14 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_teacher_exp_9/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/alldomain_r50ibn_0001/model_best.pth' --id_weight 1000 --num_epochs 120 --lr_step '80, 100'
#cd ..

# 15. pre mix, id weight * 1000, ep 120, new teacher
#cd src
#python3 train_KD.py --exp_id ps_exp_14 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_teacher_exp_9/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/pretrained/teachers/alldomain_r50ibn_0001/model_best.pth' --id_weight 1000 --num_epochs 120 --lr_step '80, 100'
#cd ..

# 16. prw reid
#cd src
#python3 train_KD.py --exp_id ps_exp_prw --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config_9000.yaml' --model_t_path '/home/helingxiao3/wei.zhang/teacher_prw_reid.pth' --id_weight 1000 --num_epochs 120 --lr_step '80, 100'
#cd ..

# 17. prw reid v2
#cd src
#python3 train_KD.py --exp_id ps_exp_prwv2 --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config_9000.yaml' --model_t_path '/home/helingxiao3/wei.zhang/teacher_prw_reid.pth' --num_epochs 60 --lr_step '50'
#cd ..

# 16. prw reid resume
#cd src
#python3 train_KD.py --exp_id ps_exp_prw --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --load_model '../exp/kd/new_teacher_exp_1/model_last.pth' --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config_9000.yaml' --model_t_path '/home/helingxiao3/wei.zhang/teacher_prw_reid.pth' --id_weight 1000 --num_epochs 120 --lr_step '80, 100'
#cd ..
#cd src
#python3 train_KD.py --exp_id ps_exp_prw --gpus 0,1,2,3 --batch_size 16 --data_cfg '../src/lib/cfg/prw.json' --kd_loss 'smooth_l1' --id_head 'base' --origin_crop True --resume --new_teacher True --reid_dim 2048 --new_teacher_cfg 'test_person_search/teacher_config.yaml' --model_t_path '/export/wei.zhang/teacher_prw_reid.pth' --id_weight 1000 --num_epochs 120 --lr_step '80, 100'
#cd ..

# 6.5H
