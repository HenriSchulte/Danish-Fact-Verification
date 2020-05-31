# Contents

This folder contains the fact verification pipeline. It contains the following files:
- classifier.py: Classifier component in Python
- config.json: Configuration file for the classifier. See CONFIG section for details
- DocumentRetrieval.jar: Compiled document retrieval component in Java
- pipeline.bat: Batch script to run pipeline on Windows. See RUN ON WINDOWS for details
- pipeline.sh: Shell script to run pipeline on MacOS. See RUN ON MAC for details


# Config

Relevant settings in config.json:
- max_len: Length of input sequences (in tokens)
- batch_size: Size of test batches
- pretrained-model: Path to the directory of the finetuned model
- enable_cuda: Whether to use CUDA for GPU-accelleration (0/1)
- enable_test_export: Whether to output predictions as a csv file (0/1)
- The remaining settings are only relevant to finetuning the model and are not used in the pipeline


# Run on Windows

The pipeline can be run in two ways, either by entering a claim directly in the command line or by providing a TSV-file containing claims and labels.
Before running the pipeline, ensure that your Python environment contains the necessary libraries listed in requirements.txt.
Additionally, you must provide the classifier with a fine-tuned model and add the path in config.json. Our fine-tuned model is available [here](https://1drv.ms/u/s!ApeFSUYI66oql9JGAX25hVLTszcD_w?e=iVIYm5). Ensure that all directories referenced in config.json exist.

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

# Run on Mac

The pipeline can be run in two ways, either by entering a claim directly in the command line or by providing a TSV-file containing claims and labels.
Before running the pipeline, ensure that your Python environment contains the necessary libraries listed in requirements.txt.
Additionally, you must provide the classifier with a fine-tuned model and add the path in config.json. Our fine-tuned model is available [here](https://1drv.ms/u/s!ApeFSUYI66oql9JGAX25hVLTszcD_w?e=iVIYm5). Ensure that all directories referenced in config.json exist.

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
