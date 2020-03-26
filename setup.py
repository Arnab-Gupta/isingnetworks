from setuptools import setup

with open ("README.md",  "r") as fh:
	long_description = fh.read()

setup(
	name='isingnetworks',
	version='1.0.2',
	description='This package is the implementation of the Ising Model on complex networks.',
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/Arnab-Gupta/ising-networks",
	author="Arnab Gupta",
	author_email="guptaarnab639@gmail.com",
	py_modules=["isingnetworks"],
	package_dir={'':'src'},
)