import argparse
import os
import sys
import torch
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.boilerplate import get_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def exp(model, input_shape, repetitions):
    input_shape = (repetitions,) + input_shape
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
    timings = np.zeros((repetitions, 1))
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            start = time.time()
            _ = model(torch.unsqueeze(dummy_input[rep], dim=0))
            curr_time = time.time() - start
            timings[rep] = curr_time
    return timings



def benchmark_model_inference(model, input_shape):

    warmup_times = 200
    experiment_times = 200
    model.to(device)

    while(True):
        warmup_timings = exp(model, input_shape, warmup_times)
        experiment_timings = exp(model, input_shape, experiment_times)
        avg, std = np.average(experiment_timings), np.std(experiment_timings)
        if std < avg/5:
            break
        else:
            print("Unstable experiment -- Rerunning...")

    print(f"{round(1000*avg, 1)} ms")
    return warmup_timings, experiment_timings

def main():
    model = get_model(num_classes=10, split_position=5, bottleneck_ratio=0.5)

    encoder = model.module.encoder
    decoder = model.module.decoder

    input_shape = (3, 96, 96)

    print("Full Model")
    benchmark_model_inference(model=model, input_shape=input_shape)

    print("Encoder Model")
    benchmark_model_inference(model=encoder, input_shape=input_shape)

    print("Decoder Model")
    input_image = torch.rand(input_shape).to(device)
    shape = encoder(torch.unsqueeze(input_image, dim=0)).shape

    benchmark_model_inference(model=decoder, input_shape=shape[1:])

    print(shape)
    print(shape[0]*shape[1]*shape[2]*shape[3])



if __name__ == "__main__":
    main()

