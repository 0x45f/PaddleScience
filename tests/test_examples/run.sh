#!/bin/bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cases=`find . -maxdepth 1 -name "test_*.py" | sort `
ignore=""
bug=0

echo "examples bug list:" >  result.txt
for file in ${cases}
do
echo ${file}
if [[ ${ignore} =~ ${file##*/} ]]; then
    echo "skip"
else
    python3.7 -m pytest ${file}
    if [ $? -ne 0 ]; then
        echo ${file} >> result.txt
        bug=`expr ${bug} + 1`
    fi
fi
done

echo "total bugs: "${bug} >> result.txt
cat result.txt
exit ${bug}
