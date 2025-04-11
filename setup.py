from setuptools import setup, find_packages

setup(
    name="deep_sort",
    version="1.0.0",
    description="Simple Online and Realtime Tracking with a Deep Association Metric",
    author="Nicolai Wojke",
    author_email="",
    url="https://github.com/nwojke/deep_sort",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 