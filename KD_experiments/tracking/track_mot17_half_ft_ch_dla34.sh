#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

#cd src
#python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34_sl1/model_last.pth --conf_thres 0.4 --val_mot17 True
#cd ..

cd src
python track_half.py kd --load_model /export/wei.zhang/PycharmProjects/FairMOT/exp/kd/kd_mot17_half_ft_ch_dla34_rh/model_last.pth --conf_thres 0.4 --val_mot17 True
cd ..