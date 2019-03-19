#!/bin/bash
bash setup/test_mini_split.sh setup/Dockerfile_3.6
exit_code_3_6="$?"

bash setup/test_mini_split.sh setup/Dockerfile_3.7
exit_code_3_7="$?"

if [ ${exit_code_3_6} -ne 0 ]; then
    echo "Failed with Python 3.6 environment"
    exit 1
elif [ ${exit_code_3_7} -ne 0 ]; then
    echo "Failed with Python 3.7 environment"
    exit 1
else
    echo "Passed all unit tests across Docker images."
fi