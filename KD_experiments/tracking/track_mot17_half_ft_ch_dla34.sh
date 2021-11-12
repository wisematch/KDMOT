#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34_sl1/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34_rh/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34_oc/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34_oc_rh/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'res'
#cd ..

#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34_oc_res/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34_oc_res_mse/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34_oc_mse/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

# track half CH_sl1 mot17 val
#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_crowdhuman_dla34/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

# track full CH_sl1 mot17 test and val
#cd src
#python track.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_crowdhuman_dla34/model_last.pth --conf_thres 0.4 --val_mot17 True #--test_mot17 True
#cd ..

# track half CH_sl1 mot17 res_head val
#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_crowdhuman_dla34_res_sl1/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'res'
#cd ..

# track half CH_sl1 mot17 val 256input res head
cd src
python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34/model_last.pth --conf_thres 0.4 --val_mot17 True --id_head 'res'
cd ..
