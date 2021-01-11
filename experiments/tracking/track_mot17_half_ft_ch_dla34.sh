cd src
python track_half.py mot --load_model ../exp/mot/mot17_half_ft_ch_dla34.pth --conf_thres 0.4 --val_mot17 True
cd ..