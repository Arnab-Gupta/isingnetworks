from setuptools import setup

with open ("README.md",  "r") as fh:
	long_description = fh.read()

setup(
	name='ising-networks',
	version='1.0.1',
	description='This package is the implementation of the Ising Model on complex networks.',
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/Arnab-Gupta/ising-networks",
	author="Arnab Gupta",
	author_email="guptaarnab639@gmail.com",
	py_modules=["ising-networks"],
	package_dir={'':'src'},
	classifiers=[
		"Programming Language :: Python 3",
		"License :: MIT License",
		"Operating system :: OS Independent"
	]
)