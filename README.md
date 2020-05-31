# Danish-Fact-Verification
This repository contains the source code and data set for an end-to-end Fact Verification System for textual claims in Danish, developed by Henri Schulte and Julie Binau as part of our MSc in Software Development at the IT University of Copenhagen.

# Data
This directory contains our data set split into training, development and test sets. The full data is available as JSONL files, including the following features:
- claim: The claim to be verified
- entity: List of page titles from which the evidence originates
- evidence: List of evidence sentences
- label: Supported, Refuted or NotEnoughInfo

For use in the pipeline, development and test sets are also available as TSV files without evidence.

# Pipeline
This directory contains our end-to-end fact verification system, consisting of an evidence retrieval and a classifier component.
- classifier.py: Classifier component in Python
- config.json: Configuration file for the classifier. See CONFIG section for details
- DocumentRetrieval.jar: Compiled document retrieval component in Java
- pipeline.bat: Batch script to run pipeline on Windows. See RUN ON WINDOWS for details
- pipeline.sh: Shell script to run pipeline on MacOS. See RUN ON MAC for details

## CONFIG

Relevant settings in config.json:
- max_len: Length of input sequences (in tokens)
- batch_size: Size of test batches
- pretrained-model: Path to the directory of the finetuned model
- enable_cuda: Whether to use CUDA for GPU-accelleration (0/1)
- enable_test_export: Whether to output predictions as a csv file (0/1)

The remaining settings are only relevant to fine-tuning the model and are not used in the pipeline.

## RUN ON WINDOWS

The pipeline can be run in two ways, either by entering a claim directly in the command line or by providing a TSV-file containing claims and labels.
Before running the pipeline, ensure that your Python environment contains the necessary libraries listed in requirements.txt.
Additionally, you must provide the classifier with a fine-tuned model and add the path in config.json. Our fine-tuned model is available at: TBD.

Entering claims directly:
1. Open the command line and navigate to the current directory
2. Run "pipeline -c"
3. Enter a claim when prompted
4. Enter the corresponding label when prompted (Refuted, Supported or NotEnoughInfo)
5. Once the pipeline has predicted a label, you will be shown a confusion matrix.
  If "enable_test_export" is turned on in the config, you will also find the prediction in the "out" directory.

Using a file of claims:
1. Open the command line and navigate to the current directory
2. Run "pipeline -f [path] [k] [l]" where the parameters are replaced by actual values:
    - [path] is the path to the claims file (e.g. "../data/tsv/dev.tsv")
    - [k] is the number of articles from which to retrieve evidence
    - [l] is the number of evidence sentences to retrieve
3. Once the pipeline has predicted labels for all claims, you will be shown a confusion matrix.
  If "enable_test_export" is turned on in the config, you will also find the predictions in the "out" directory.

## RUN ON MAC

The pipeline can be run in two ways, either by entering a claim directly in the command line or by providing a TSV-file containing claims and labels.
Before running the pipeline, ensure that your Python environment contains the necessary libraries listed in requirements.txt.
Additionally, you must provide the classifier with a fine-tuned model and add the path in config.json. Our fine-tuned model is available at: TBD.

Entering claims directly:
1. Open the command line and navigate to the current directory
2. Run "./pipeline -c"
3. Enter a claim when prompted
4. Enter the number corresponding the label when prompted (0=Refuted, 1=Supported or 2=NotEnoughInfo)
5. Once the pipeline has predicted a label, you will be shown a confusion matrix.
  If "enable_test_export" is turned on in the config, you will also find the prediction in the "out" directory.

Using a file of claims:
1. Open the command line and navigate to the current directory
2. Run "./pipeline -f [path] [k] [l]" where the parameters are replaced by actual values:
    - [path] is the path to the claims file (e.g. "../data/tsv/dev.tsv")
    - [k] is the number of articles from which to retrieve evidence
    - [l] is the number of evidence sentences to retrieve
3. Once the pipeline has predicted labels for all claims, you will be shown a confusion matrix.
  If "enable_test_export" is turned on in the config, you will also find the predictions in the "out" directory.

# Source Code

This directory contains the source code for our data generation tools, the classifier and evidence retrieval.

## Data Generation Tools

This folder contains the source code to our data generation tools. We omit data files and intermediate results.
All of these tools may be run from the command line without any parameters. For the sake of simplicity, we hardcode file paths, which may no longer be accurate.
When stepping through our data generation workflow, we use the following order:

1. Wiki Article Parser
2. Claim Generation Tool
3. Mutation Generation Tool
4. Dictionary Generation Tool
5. Annotation Tool

The Translator relies only on the Wiki Article Parser.

## Classifier

The classifier can be run for either fine-tuning (training), testing or both. To change the configuration, edit the config.json file in the same directory.
The config's items are explained below. Settings marked with * only affect fine-tuning. Elements marked with + only affect testing.
Disclaimer: Not all settings are compatible with each other, and incorrect settings will crash the program!

- max_len: Length of input sequences (in tokens)
- batch_size: Size of test batches, decrease in case of out-of-memory issues
- num_labels: Number of labels in the data, in our case always 3
- num_epochs*: Number of fine-tuning epochs
- learning_rate*: Learning rate for fine-tuning
- train_data_path*: Path to train data as jsonl
- validation_data_path*: Path to validation data as jsonl
- test_data_path+: Path to test data as jsonl
- enable_test: Whether to test the model on the test data (0/1)
- data_sample: Allows for random sampling of the data. 1 uses all provided data (100%), 0.5 uses (50%), etc.
- pretrained-model: Path to the directory of the pre-trained model or name of a BERT model (e.g. bert-base-multilingual-cased)
- model_save_dir*: Directory in which to save the fine-tuned model
- enable_plotting: Whether to enable plotting of loss and confusion matrix (0/1)
- enable_training: Whether to fine-tune the model (0/1)
- output_dir*: Directory in which to export training results
- enable_cuda: Whether to use CUDA for GPU-accelleration (0/1)

Currently, config.json is configured for fine-tuning a model on a Tesla K80. batch_size and/or max_len need to be reduced for GPUs with less memory.

## Evidence Retrieval

The behaviour of our document retrieval component can be affected through its parameters:

1. No parameters: The program will look for a file "claims.tsv" in the same directory and for each claim it contains retrieve 2 evidence sentences from the 5 most relevant documents.
2. Claim and label: The program will take the provided claim and retrieve 2 evidence sentences from the 5 most relevant documents. The label is just provided to be included in the output.
3. file path, k and l: The program will look for a TSV file at the provided path and for each claim it contains retrieve l evidence sentences from the k most relevant documents.

In order to run, the program requires Apache Lucene and Apache OpenNLP. Additionally, it excepts OpenNLP models for tokenizing and POS-tagging. Danish models are available at http://opennlp.sourceforge.net/models-1.5/.
