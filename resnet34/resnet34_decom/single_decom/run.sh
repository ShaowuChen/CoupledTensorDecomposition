python single_tt_decom_net.py --gpu=0 --rank_rate=3/8 --train_set_number_rate=0.5 --num_lr=1e-3
python single_tt_decom_net.py --gpu=0 --rank_rate=1/2 --train_set_number_rate=0.5 --num_lr=1e-3
python single_tt_decom_net.py --gpu=0 --rank_rate=3/4 --train_set_number_rate=0.5 --num_lr=1e-4





python single_tt_decom_net.py --gpu=7 --rank_rate=3/4 --train_set_number_rate=1 --num_lr=1e-3 --epoch=20


python rerun.py --gpu=0 --rank_rate=3/4 --train_set_number_rate=1 --num_lr=1e-4 --epoch=6
