
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lenyabloko/SemEval2020/blob/master/SemEval2020_Task5_Subtask_2.ipynb)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ethan-at-semeval-2020-task-5-modelling-causal/counterfactual-detection-on-semeval-2020-2)](https://paperswithcode.com/sota/counterfactual-detection-on-semeval-2020-2?p=ethan-at-semeval-2020-task-5-modelling-causal)

This code was tested on Google Cloud Compute Engine N1 high memory (26G) instance of Linux Ubuntu 18.04 with added GPU Tesla V100 running Cuda (for help on adding GPU to your Cloud instance see this SO question: https://stackoverflow.com/questions/53415180/gcp-error-quota-gpus-all-regions-exceeded-limit-0-0-globally)

To run test code you will need to install some basic software like Pip3 (https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/) and Cuda (https://gist.github.com/ingo-m/60a21120f3a9e4d7dd1a36307f3a8cce):

```
sudo apt update
sudo apt install python3-pip
...

```

After Cuda installed you should execute the setup.sh with will install all required packages

```
$ sudo sh ./setup.sh

```

Then, you must create custom directories for input and output files and place your train.csv and test.csv into the ./content 

```
$ sudo mkdir outputs
$ sudo mkdir content

$ sudo cp train.csv /content
$ sudo cp test.csv /content

```

Finally, you can run the semeval.py (you should see the following in the console):

```
$ python3 semeval.py
cpu
2
INFO:pytorch_transformers.modeling_utils:loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json from cache at /home/leonidyabloko/.cache/torch/pytorch_transformers/e1a2a406b5a05063c31f4dfdee7608986ba7c6393f7f79db5e69dcd197208534.a7ab0e5de2d8321d6d6a15b199110f2c99be72976b7d151423cb8d8c261a13b6
INFO:pytorch_transformers.modeling_utils:Model config {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": "binary",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "pruned_heads": {},
  "torchscript": false,
  "type_vocab_size": 1,
  "vocab_size": 50265
}

```

When citing ETHAN in academic papers and theses, please use this BibTeX entry:

```
@inproceedings{yabloko-2020-ethan,
    title = "{ETHAN} at {S}em{E}val-2020 Task 5: Modelling Causal Reasoning in Language Using Neuro-symbolic Cloud Computing",
    author = "Yabloko, Len",
    booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
    month = dec,
    year = "2020",
    address = "Barcelona (online)",
    publisher = "International Committee for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.semeval-1.83",
    pages = "645--652",
    abstract = "I present ETHAN: Experimental Testing of Hybrid AI Node implemented entirely on free cloud computing infrastructure. The ultimate goal of this research is to create modular reusable hybrid neuro-symbolic architecture for Artificial Intelligence. As a test case I model natural language comprehension of causal relations from open domain text corpus that combines semi-supervised language model (Huggingface Transformers) with constituency and dependency parsers (Allen Institute for Artificial Intelligence.)",
}
```


