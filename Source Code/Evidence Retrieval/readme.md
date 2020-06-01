The behaviour of our document retrieval component can be affected through its parameters:

- No parameters: The program will look for a file "claims.tsv" in the same directory and for each claim it contains retrieve 2 evidence sentences from the 5 most relevant documents.
- Claim and label: The program will take the provided claim and retrieve 2 evidence sentences from the 5 most relevant documents. The label is just provided to be included in the output.
- File path, k and l: The program will look for a TSV file at the provided path and for each claim it contains retrieve l evidence sentences from the k most relevant documents.

In order to run, the program requires Apache Lucene and Apache OpenNLP. Additionally, it excepts OpenNLP models for tokenizing and POS-tagging. Danish models are available at http://opennlp.sourceforge.net/models-1.5/.

If you like good Java code: I am sorry.
