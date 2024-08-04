from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
	name='scouter-learn',
	version='0.1.2',
	description='A transcriptional response predictor for unseen genetic perturbtions with LLM embeddings',
    long_description=README,
    long_description_content_type='text/markdown',
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