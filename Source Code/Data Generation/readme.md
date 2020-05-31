This folder contains the source code to our data generation tools. We omit data files and intermediate results.
All of these tools may be run from the command line without any parameters. For the sake of simplicity, we hardcode file paths, which may no longer be accurate.

When stepping through our data generation workflow, we use the following order:

1. Wiki Article Parser
2. Claim Generation Tool
3. Mutation Generation Tool
4. Dictionary Generation Tool
5. Annotation Tool

The Translator relies only on the Wiki Article Parser.
