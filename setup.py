from setuptools import setup, find_namespace_packages


setup(
    name="earthcare-earlinet-level1",
    version="0.1.0",
    description="Utilities for working with EarthCARE EARLINET Level 1 data",
    long_description=(
        "Installable package containing a full EARLINET - EarthCARE ATL_NOM_1B comparison utilities module. "
        "Includes processing and analysis helpers used in level_1_comparer.py to compare scattering ratios from the two LIDAR systems."
    ),
    long_description_content_type="text/plain",
    author="",
    packages=find_namespace_packages(include=["earthcare_earlinet_level1*"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "xarray",
        "matplotlib",
        "scipy",
        "geopy",
        "loguru",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
