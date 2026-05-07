from setuptools import setup, find_packages

setup(
    name="disco-ttg",
    version="0.1.0",
    description="DISCO: Test-Time Generalization via Neural Operator Splitting",
    author="Louis Serrano",
    packages=find_packages(exclude=["GEPS", "GEPS.*", "MPP", "MPP.*", "ZEBRA", "ZEBRA.*",
                                     "notebooks", "notebooks.*", "tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "lightning>=2.0",
        "numpy",
        "einops",
        "hydra-core",
        "omegaconf",
        "wandb",
        "torchode",
        "tqdm",
        "h5py",
    ],
)