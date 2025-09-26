# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


gitCloneAndApplyPatch() {
    url=$1
    foldName=$2
    tag=$3
    patch=$4
    if [ ! -d "$foldName" ]; then
        git clone $url
        cd $foldName
        git checkout $tag
        cp ../mmdetection/diff_patch/$patch ./
        git apply $patch
    else
        cd $foldName
    fi
}

main() {
    # 安装基础依赖
    pip install -r demo/requirements.txt
    pip install -U openmim
    cd ..
    # 安装mmengine
    gitCloneAndApplyPatch https://github.com/open-mmlab/mmengine.git mmengine v0.10.6 mmengine_diff.patch
    pip install -e .
    cd ..
    # 安装mmcv
    gitCloneAndApplyPatch https://github.com/open-mmlab/mmcv.git mmcv v2.1.0 mmcv_diff.patch
    pip install -r requirements/optional.txt
    pip install -e .
    cd ..
    # 安装mmdetection
    cd mmdetection
    git checkout cfd5d3a9
    cp diff_patch/mmdetection_diff.patch ./
    git apply mmdetection_diff.patch
    pip install -e .
}

main
