cd src
python track_half.py mot --load_model ../exp/mot/crowdhuman_dla34.pth --conf_thres 0.4 --val_mot17 True
cd ..