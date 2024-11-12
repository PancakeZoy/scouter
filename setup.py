from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

# Read the version from scouter/_version.py
def read_version():
    version_dict = {}
    with open(HERE / "scouter" / "_version.py") as version_file:
        exec(version_file.read(), version_dict)
    return version_dict['__version__']

setup(
	name='scouter-learn',
	version=read_version(),
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
		'scipy >= 1.14.0',
        'requests >= 2.0.0'
	],
	python_requires='>=3.10.0'
)