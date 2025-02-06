import os
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isdir, isfile, join
#%matplotlib inline

sz = "124M"

loss_baseline = {
    "124M": 3.2924,
}[sz]
hella2_baseline = { # HellaSwag for GPT-2
    "124M": 0.294463,
    "350M": 0.375224,
    "774M": 0.431986,
    "1558M": 0.488946,
}[sz]
hella3_baseline = { # HellaSwag for GPT-3
    "124M": 0.337,
    "350M": 0.436,
    "774M": 0.510,
    "1558M": 0.547,
}[sz]
root = 'log'
models = [f for f in os.listdir(root) if isdir(join(root, f))]
# load the log file
for model_type in models:
    model_log_path = join(root, model_type)
    params = [f for f in os.listdir(model_log_path) if isfile(join(model_log_path, f))]
    params = [param for param in params if "log" in param]
    for param in params:
        param_count = param.split('.')[0].split('_')[1]
        with open(f"log/{model_type}/{param}", "r") as f:
            lines = f.readlines()

        # parse the individual lines, group by stream (train,val,hella)
        streams = {}
        for line in lines:
            step, stream, val = line.strip().split()
            if stream not in streams:
                streams[stream] = {}
            streams[stream][int(step)] = float(val)

        # convert each stream from {step: val} to (steps[], vals[])
        # so it's easier for plotting
        streams_xy = {}
        for k, v in streams.items():
            # get all (step, val) items, sort them
            xy = sorted(list(v.items()))
            # unpack the list of tuples to tuple of lists
            streams_xy[k] = list(zip(*xy))

        # create figure
        plt.figure(figsize=(16, 6))

        # Panel 1: losses: both train and val
        plt.subplot(121)
        # horizontal line at GPT-2 baseline
        if loss_baseline is not None:
            plt.axhline(y=loss_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint val loss")
        xs, ys = streams_xy["train"] # training loss
        ys = np.array(ys)
        plt.plot(xs, ys, label=f'nano{model_type.lower()} ({param_count}) train loss')
        print("Min Train Loss:", min(ys))
        xs, ys = streams_xy["val"] # validation loss
        plt.plot(xs, ys, label=f'nano{model_type.lower()} ({param_count}) val loss')
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.yscale('log')
        plt.ylim(top=6.0)
        plt.legend()
        plt.title("Loss")
        print("Min Validation Loss:", min(ys))

        # Panel 2: HellaSwag eval
        plt.subplot(122)
        # horizontal line at GPT-2 baseline
        if hella2_baseline:
            plt.axhline(y=hella2_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint")
        if hella3_baseline:
            plt.axhline(y=hella3_baseline, color='g', linestyle='--', label=f"OpenAI GPT-3 ({sz}) checkpoint")
        xs, ys = streams_xy["hella"] # HellaSwag eval
        ys = np.array(ys)
        plt.plot(xs, ys, label=f"nano{model_type.lower()} ({param_count})")
        plt.xlabel("steps")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("HellaSwag eval")
        print("Max Hellaswag eval:", max(ys), "\n")
        plt.savefig(f'log/{model_type}/loss_{param_count}.png')
        plt.clf()

# create figure
plt.figure(figsize=(16, 6))
draw_baseline = True
for model_type in models:
    model_log_path = join(root, model_type)
    params = [f for f in os.listdir(model_log_path) if isfile(join(model_log_path, f))]
    params = [param for param in params if "log" in param]
    for param in params:
        param_count = param.split('.')[0].split('_')[1]
        with open(f"log/{model_type}/{param}", "r") as f:
            lines = f.readlines()

        # parse the individual lines, group by stream (train,val,hella)
        streams = {}
        for line in lines:
            step, stream, val = line.strip().split()
            if stream not in streams:
                streams[stream] = {}
            streams[stream][int(step)] = float(val)

        # convert each stream from {step: val} to (steps[], vals[])
        # so it's easier for plotting
        streams_xy = {}
        for k, v in streams.items():
            # get all (step, val) items, sort them
            xy = sorted(list(v.items()))
            # unpack the list of tuples to tuple of lists
            streams_xy[k] = list(zip(*xy))

        # Panel 1: losses: both train and val
        plt.subplot(121)
        # horizontal line at GPT-2 baseline
        if loss_baseline is not None and draw_baseline:
            plt.axhline(y=loss_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint val loss")
        xs, ys = streams_xy["val"] # validation loss
        plt.plot(xs, ys, label=f'nano{model_type.lower()} ({param_count}) val loss')
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.yscale('log')
        plt.ylim(top=4.0)
        plt.legend()
        plt.title("Loss")

        # Panel 2: HellaSwag eval
        plt.subplot(122)
        # horizontal line at GPT-2 baseline
        if hella2_baseline and draw_baseline:
            plt.axhline(y=hella2_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint")
        if hella3_baseline and draw_baseline:
            plt.axhline(y=hella3_baseline, color='g', linestyle='--', label=f"OpenAI GPT-3 ({sz}) checkpoint")
            draw_baseline = False
        xs, ys = streams_xy["hella"] # HellaSwag eval
        ys = np.array(ys)
        plt.plot(xs, ys, label=f"nano{model_type.lower()} ({param_count})")
        plt.xlabel("steps")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("HellaSwag eval")
plt.savefig(f'fig/loss_all.png')
plt.clf()
