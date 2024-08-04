from setuptools import setup, find_packages

setup(
	name='scouter',
	version='0.1.0',
	description='A transcriptional response predictor for unseen genetic perturbtions with LLM embeddings',
	packages=find_packages(include=['scouter', 'scouter.*']),
	author='Ouyang Zhu, Jun Li',
	author_email='ozhu@nd.edu',
	url='https://github.com/PancakeZoy/scouter',
	install_requires=[
		'torch >= 2.0.0',
		'tqdm >= 4.0.0',
		'anndata >= 0.10.0',
		'pandas >= 2.2.0',
		'numpy >= 1.26.0',
		'scanpy >= 1.10.0',
		'seaborn >= 0.13.0',
		'matplotlib >= 3.9.0',
		'scikit-learn >= 1.5.0',
		'scipy >= 1.14.0'
	],
	python_requires='>=3.11.0'
)