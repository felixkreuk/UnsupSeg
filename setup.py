import setuptools

__VERSION__ = "0.0.1"

setuptools.setup(
    name="unsup_seg",
    version=__VERSION__,
    author="Max Kraan <max@klay.vision>",
    description="Unsupervised Phoneme Segmentation",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.7.13",
)
