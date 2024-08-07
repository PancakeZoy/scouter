<p align="left">
  <img src="https://github.com/PancakeZoy/scouter/blob/master/img/ScouterLogo.png?raw=true" width="100" title="logo">
</p>

# Scouter: a transcriptional response predictor for unseen genetic perturbtions with LLM embeddings

Scouter is a deep neural network with simple architecture, for the task of predicting transcriptional response to unseen genetic perturbtions.

Scouter employs the LLM embeddings generated from text description of genes, enabling the perdiction on unseen genes.

For more details read our [manuscript]()
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
from scouter import Scouter, ScouterData

# please prepare the gene expression data (adata) and gene embedding dataframe (embd)
pertdata = ScouterData(adata=adata, embd=embd)
pertdata.setup_ad('embd_index')

# Model Training
scouter_model = Scouter(pertdata)
scouter_model.model_init()
scouter_model.train()

# Prediction
scouter_model.precit('GeneA+ctrl')
```

## Demos

| Name | Description |
|-----------------|-------------|
| [Model Tutorial](demo/ModelTutorial.ipynb) | A detailed tutorial on how to use Scouter on Adamson dataset, including preprocessing, paramter setting, model evaluation|
| [Unmatched Genes Tutorial](demo/UnmatchRemedy.ipynb) | A tutorial that illustrates the problem of unmatched genes between adata and embd, and provides a remedy.|