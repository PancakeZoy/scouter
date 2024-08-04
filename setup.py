from setuptools import setup

setup(
	name='scouter',
	version='0.1',
	description='Gene Perturbation Prediction with LLM',
	author='Ouyang Zhu, Jun Li',
	author_email='ozhu@nd.edu',
	url='https://github.com/PancakeZoy/scouter',
	install_requires=[
		'torch >= 2.0.0',
		'tqdm',
		'anndata',
		'pandas',
		'numpy',
		'scanpy',
		'seaborn',
		'matplotlib',
		'scikit-learn',
		'scipy'
	]
)