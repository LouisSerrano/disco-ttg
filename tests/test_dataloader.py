import time
import torch
from torch.utils.data import DataLoader
from train import TemporalBatchDatasetFly

if __name__ == "__main__":
    # Parameters for the test
    n_batches = 50
    batch_size = 256
    sub_x = 1
    sub_t = 1
    input_frames = 16
    output_frames = 16
    L = 16.0
    nx = 256
    nt = 100
    T = 10.0
    v_range = (0.01, 1.0)
    D_range = (0.01, 1.0)
    fractal_degree = 8
    fractal_power_range = (1.0, 8.0)
    seed = 42

    dataset = TemporalBatchDatasetFly(
        n_batches=n_batches,
        batch_size=batch_size,
        sub_x=sub_x,
        sub_t=sub_t,
        input_frames=input_frames,
        output_frames=output_frames,
        L=L,
        nx=nx,
        nt=nt,
        T=T,
        v_range=v_range,
        D_range=D_range,
        fractal_degree=fractal_degree,
        fractal_power_range=fractal_power_range,
        seed=seed,
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=4)

    start = time.time()
    times = []
    n_test_batches = 20
    print(f"Testing speed for {n_test_batches} batches...")
    for i, batch in enumerate(dataloader):
        if i >= n_test_batches:
            break
        print(batch['input'].shape)
        # Fetch batch (already done by DataLoader)
        # Optionally, move to device or do something with batch
    end = time.time()
    print(f"Average batch fetch time: {(end - start)/n_test_batches} seconds") 