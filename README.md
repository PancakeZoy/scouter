<p align="left">
  <img src="https://github.com/PancakeZoy/scouter/blob/master/img/ScouterLogo.png?raw=true" width="100" title="logo">
</p>

# Scouter: a transcriptional response predictor for unseen genetic perturbtions with LLM embeddings

Scouter is a deep neural network with simple architecture, for the task of predicting transcriptional response to unseen genetic perturbtions.

Scouter employs the LLM embeddings generated from text description of genes, enabling the perdiction on unseen genes.

For more details read our [manuscript]().

Code for reproducing results: [Link](https://github.com/PancakeZoy/scouter_misc).

<p align="center">
  <img src="https://github.com/PancakeZoy/scouter/blob/master/img/workflow_horizontal.png?raw=true" width="750" title="logo">
</p>
<br>
<p align="center">
  <img src="https://github.com/PancakeZoy/scouter/blob/master/img/scouter_horizontal.png?raw=true" width="750" title="logo">
</p>

## Installation
`pip install scouter-learn`

## Main API
Below is an example that includes main APIs to train `Scouter` on a perturbation dataset. 

```python
from scouter import Scouter, ScouterData, adamson_small, embedding_small

adata = adamson_small()
embd = embedding_small()
scouterdata = ScouterData(adata=adata, embd=embd, key_label='condition', key_var_genename='gene_name')
scouterdata.setup_ad('embd_index')
scouterdata.gene_ranks()
scouterdata.get_dropout_non_zero_genes()
scouterdata.split_Train_Val_Test(seed=1)

# Model Training
scouter_model = Scouter(scouterdata)
scouter_model.model_init()
scouter_model.train()

# Prediction
scouter_model.pred(['ATP5B+ctrl', 'MANF+ctrl'])

# Evaluation
scouter_model.barplot('MANF+ctrl')
```

## Demos

| Name | Description |
|-----------------|-------------|
| [Demo.ipynb](demo/Demo.ipynb) | A detailed tutorial on how to apply `Scouter` to a smaller version of Adamson dataset, including preprocessing, paramter setting, model training, and evaluation|
| [OwnDataTutorial.ipynb](demo/OwnDataTutorial.ipynb)| A tutorial to guide users to load their own dataset and embedding matrix.
| [UnmatchRemedy.ipynb](demo/UnmatchRemedy.ipynb) | A tutorial that illustrates the problem of unmatched genes between perturbation dataset and embedding matrix, and provides a remedy.|