#!/bin/sh
DIR="$( cd "$( dirname $0 )" && pwd )"

git_hash=$(git log --pretty=format:"%h" -1)
py_file=${1}
if [ "$#" != "2" ]; then
    csv_file=${py_file/.py/.${git_hash}.csv}
    log_file=${py_file/.py/.${git_hash}.log}
else
    csv_file=${py_file/.py/.${2}.csv}
    log_file=${py_file/.py/.${2}.log}
fi
echo "python ${py_file} ${csv_file}"

python ${py_file} ${csv_file} | tee -a ${DIR}/output/${log_file}
