docker run -it --rm --gpus 1 -v /data2/nihansen/code/dreamerv3:/code img:latest /bin/bash -c "cd /code && python train.py task=cartpole-balance seed=1"
