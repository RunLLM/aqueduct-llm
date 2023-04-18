from setuptools import setup, find_packages

install_requires = open("requirements.txt").read().strip().split("\n")

setup(
    name="aqueduct-llm",
    version="0.0.0",
    install_requires=install_requires,
    packages=find_packages(),
)
