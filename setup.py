from setuptools import setup, find_packages

setup(
    name="rl_asset_allocation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pytest"
    ],
    python_requires=">=3.7"
)