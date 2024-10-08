# Detection of Text Generated by a Large Language Model

## Abstract

Large language models (LLMs) are used for problems in the field of natural language processing such as text classification, generating or summarizing text and others. The misuse of LLMs created a need for systems which would be capable of detecting texts generated by an LLM among texts written by humans. For solving this problem, classifiers BERT, RoBERTa and DistilBERT were implemented and trained using the "GPT Wiki Intro" dataset. A share of the texts in the training dataset was generated by the Curie LLM. After the classifiers were trained, their predictions on the testing dataset were validated using various metrics. The classifiers were additionally validated using texts that were generated by different LLMs.

**Keywords:** natural language processing; large language models; text classification; BERT; RoBERTa; DistilBERT; transformer models; confusion matrix