# How are Prompts Different in Terms of Sensitivity?

This repository includes the code for the paper *How are Prompts Different in Terms of Sensitivity?*. You will find codes here to estimate sensitivity, calculate gradient-based saliency scores, and utilize sensitivity-aware decoding.
 
> **Abstract:** In-context learning (ICL) has become one of the most popular learning paradigms. While there is a growing body of literature focusing on prompt engineering, there is a lack of systematic analysis comparing the effects of prompt techniques across different models and tasks. To address this, we present a comprehensive prompt analysis based on the sensitivity of a function. Our analysis reveals that sensitivity is an unsupervised proxy for model performance, as it exhibits a strong negative correlation with accuracy. We use gradient-based saliency scores to empirically demonstrate how different prompts affect the relevance of input tokens to the output, resulting in different levels of sensitivity. Furthermore, we introduce *sensitivity-aware* decoding which incorporates sensitivity estimation as a penalty term in the standard greedy decoding. We show that this approach is particularly helpful when information in the input is scarce. Our work provides a fresh perspective on the analysis of prompts, and contributes to a better understanding of the mechanism of ICL.

Contact person: Sheng Lu

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

## Getting Started
Prepare the environment:

```python
pip install -r requirements.txt
```

## Inference
See [inference.ipynb](https://github.com/UKPLab/naacl2024-prompt-sensitivity/blob/main/inference.ipynb) for the inference code.

## Saliency scores
See [saliency.ipynb](https://github.com/UKPLab/naacl2024-prompt-sensitivity/blob/main/saliency.ipynb) for the calculation of gradient-based saliency scores ([Simonyan et al., 2013](https://arxiv.org/abs/1312.6034); [Li et al., 2016](https://aclanthology.org/N16-1082/); [Yin and Neubig, 2022](https://aclanthology.org/2022.emnlp-main.14/)).

## Sensitivity-aware decoding
See [sensitivity_aware_decoding.ipynb](https://github.com/UKPLab/naacl2024-prompt-sensitivity/blob/main/sensitivity_aware_decoding.ipynb) for the implementation of sensitivity-aware decoding.

## Evaluation scores
See [evaluation_scores.csv](https://github.com/UKPLab/naacl2024-prompt-sensitivity/blob/main/evaluation_scores.csv) and [evaluation_scores - greedy_decoding.csv](https://github.com/UKPLab/naacl2024-prompt-sensitivity/blob/main/evaluation_scores%20-%20greedy_decoding.csv) for the full evaluation scores.

## Citation
Please use the following citation:

```
@inproceedings{lu-etal-2024-prompts,
    title = "How are Prompts Different in Terms of Sensitivity?",
    author = "Lu, Sheng  and
      Schuff, Hendrik  and
      Gurevych, Iryna",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.325",
    pages = "5833--5856",
    abstract = "In-context learning (ICL) has become one of the most popular learning paradigms. While there is a growing body of literature focusing on prompt engineering, there is a lack of systematic analysis comparing the effects of prompt techniques across different models and tasks. To address this, we present a comprehensive prompt analysis based on sensitivity. Our analysis reveals that sensitivity is an unsupervised proxy for model performance, as it exhibits a strong negative correlation with accuracy. We use gradient-based saliency scores to empirically demonstrate how different prompts affect the relevance of input tokens to the output, resulting in different levels of sensitivity. Furthermore, we introduce sensitivity-aware decoding which incorporates sensitivity estimation as a penalty term in the standard greedy decoding. We show that this approach is particularly helpful when information in the input is scarce. Our work provides a fresh perspective on the analysis of prompts, and contributes to a better understanding of the mechanism of ICL.",
}
```

## Disclaimer
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
