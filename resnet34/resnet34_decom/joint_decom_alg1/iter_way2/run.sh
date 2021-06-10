python iter2.py --rank_rate_yellow=3/8 --rank_rate_ave=3/16 --rank_rate_single=0.343830232341034 --gpu=0

python iter2.py --rank_rate_yellow=3/8 --rank_rate_ave=1/4 --rank_rate_single=0.3261854184542645 --gpu=0&
python iter2.py --rank_rate_yellow=3/8 --rank_rate_ave=3/8 --rank_rate_single=0.27720918028416913 --gpu=1&
python iter2.py --rank_rate_yellow=3/8 --rank_rate_ave=1/2 --rank_rate_single=0.20261683547141943 --gpu=2&

python iter2.py --rank_rate_yellow=1/2 --rank_rate_ave=1/4 --rank_rate_single=0.4619342907094595 --gpu=3&
python iter2.py --rank_rate_yellow=1/2 --rank_rate_ave=3/8 --rank_rate_single=0.4254943552113008 --gpu=4&
python iter2.py --rank_rate_yellow=1/2 --rank_rate_ave=1/2 --rank_rate_single=0.3742526722974462 --gpu=5&

python iter2.py --rank_rate_yellow=3/4 --rank_rate_ave=1/4 --rank_rate_single=0.7234166524194108 --gpu=6&
python iter2.py --rank_rate_yellow=3/4 --rank_rate_ave=3/8 --rank_rate_single=0.6988277643866937 --gpu=7&
python iter2.py --rank_rate_yellow=3/4 --rank_rate_ave=1/2 --rank_rate_single=0.6658274693443998 --gpu=0


# r_c        r_s             r_yellow
# 3\16 0.34383023234103400    3\8     0.375
# 1\4 0.32618541845426400     3\8 
# 3\8 0.27720918028416900     3\8 
            
# 1\4 0.46193429070945900     1\2 
# 3\8 0.42549435521130000     1\2     0.5
# 1\2 0.37425267229744600     1\2 
            
# 1\4 0.72341665241941000     3\4 
# 3\8 0.69882776438669300     3\4     0.75
# 1\2 0.66582746934439900     3\4 

# r_yellow是原in维度！=out维度的层
# 但是这个目录下的该层实际上没做分解
# r_yellow只是名字上的区分；通过r_yellow可以获取iter10.npy文件，而实际上这些npy就是根据各自r_s求解出来的
# 有个目录ave=3_1，实际上是3——16
