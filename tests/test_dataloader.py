import time
import torch
from torch.utils.data import DataLoader
from train import TemporalBatchDatasetFly
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Parameters for the test
    n_batches = cfg.test.get('n_batches', 50)
    batch_size = cfg.test.get('batch_size', 256)
    sub_x = cfg.test.get('sub_x', 1)
    sub_t = cfg.test.get('sub_t', 1)
    input_frames = cfg.test.get('n_input_frames', 16)
    output_frames = cfg.test.get('n_output_frames', 16)
    L = cfg.data.get('L', 16.0)
    nx = cfg.data.get('nx', 256)
    nt = cfg.data.get('nt', 100)
    T = cfg.data.get('T', 10.0)
    v_range = tuple(cfg.data.get('v_range', [0.01, 1.0]))
    D_range = tuple(cfg.data.get('D_range', [0.01, 1.0]))
    fractal_degree = cfg.data.get('fractal_degree', 8)
    fractal_power_range = tuple(cfg.data.get('fractal_power_range', [1.0, 8.0]))
    seed = cfg.test.get('seed', 42)

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
    n_test_batches = cfg.test.get('n_test_batches', 20)
    print(f"Testing speed for {n_test_batches} batches...")
    for i, batch in enumerate(dataloader):
        if i >= n_test_batches:
            break
        print(batch['input'].shape)
        # Fetch batch (already done by DataLoader)
        # Optionally, move to device or do something with batch
    end = time.time()
    print(f"Average batch fetch time: {(end - start)/n_test_batches} seconds")

if __name__ == "__main__":
    main() 