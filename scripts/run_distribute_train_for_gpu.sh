#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

DATA_DIR="/data/imagenet/demo"
export RANK_SIZE=2
export NCCL_SOCKET_IFNAME=ens11f0
# PATH_CHECKPOINT=""
# if [ $# == 2 ]
# then
# 	PATH_CHECKPOINT=$2
# fi


mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    -H 192.168.1.49:1, 192.168.1.52:1 \
	python train.py  \
    --is_distribute=1 \
    --platform="GPU" \
    --per_batch_size=250 \
    --pretrained=$PATH_CHECKPOINT \
    --data_dir=$DATA_DIR > log.txt 2>&1 &
