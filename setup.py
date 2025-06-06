from setuptools import setup, find_packages

setup(
    name="pdecontrolgym",
    version="0.0.1",
    packages=find_packages(include=["pde_control_gym", "pde_control_gym.*"]),
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib",
        "stable-baselines3"
    ],
)