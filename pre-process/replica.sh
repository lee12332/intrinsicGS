#!/bin/bash

# root_dir="/home/lhy/research/intrinsic-gs/datasets/replica/"
# list="chair drums ficus hotdog lego materials mic ship"
# list="office_0 office_1 office_2"
# list="room_1 room_2 office_0 office_1 office_2 office_3 office_4"
list="office_3"
# 可行的scene room_0 room_1 
# 存在问题的scene room_2  office_0 office_1  No good initial image pair found
for i in $list; do
python replica.py --scene ${i}
done
