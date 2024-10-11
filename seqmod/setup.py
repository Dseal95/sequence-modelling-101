from setuptools import setup, find_packages

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name="seqmod",
    version="1.0",
    description="A package to demonstrate how LSTMs and ConvLSTMS work.",
    author="Dan Seal",
    packages=find_packages(),
    install_requires=requirements,
)
