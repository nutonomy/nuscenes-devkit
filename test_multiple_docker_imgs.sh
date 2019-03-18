#!/bin/bash
bash test_mini_split.sh Dockerfile_3.5
exit_code_3_5="$?"

bash test_mini_split.sh Dockerfile_3.6
exit_code_3_6="$?"

bash test_mini_split.sh Dockerfile_3.7
exit_code_3_7="$?"

if [ ${exit_code_3_5} -ne 0 ]; then
    echo "Failed with Python 3.5 environment"
    exit 1
elif [ ${exit_code_3_6} -ne 0 ]; then
    echo "Failed with Python 3.6 environment"
    exit 1
elif [ ${exit_code_3_7} -ne 0 ]; then
    echo "Failed with Python 3.5 environment"
    exit 1
else
    echo "Passed all unit tests across Docker images."
fi