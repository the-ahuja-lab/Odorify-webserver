<div align="center"> <h1>OdoriFy </h1>
 <b>A comprehensive AI-driven web-based solution for Human Olfaction</b>
 </div>

OdoriFy is an open-source web server with Deep Neural Network-based prediction models coupled with explainable Artificial Intelligence functionalities, developed in an effort to provide the users with a one-stop destination for all their problems pertaining to olfaction. OdoriFy is highly versatile, capable of deorphanizing human olfactory receptors (Odor Finder), odorant prediction (Odorant Predictor), identification of Responsive Olfactory Receptors for a given ligand/s (OR Finder), and Odorant-OR Pair Analysis. With OdoriFy, we aim to provide a simplified and hassle-free environment for users.
Webserver is freely available at [https://odorify.ahujalab.iiitd.edu.in/olfy/](http://odorify.ahujalab.iiitd.edu.in/olfy/)

![Odorify logo ](https://imgur.com/2gJZMWo)

## Index


## Prediction Engines:

1.  **Odorant Predictor:** OdoriFy allows users to predict or verify whether the user supplemented chemicals qualifies for the odorant properties. It also performs the sub-structure analysis and highlights atoms indispensable for the predicted decision. 
<u>Input:</u> Chemical (SMILES)
    
2.  **OR Finder**: It enables the identification of cognate human olfactory receptors for the user-supplied odorant molecules. Moreover, similar to Odorant Predictor it also highlights odorant sub-structures responsible for the predicted interactions. 
<u>Input:</u> Chemical (SMILES)

3.  **Odor Finder:** OdoriFy allows users to submit the protein sequences of wild-type or mutant human ORs and performs prediction and ranking of their potential cognate odorants.
 <u>Input:</u> Protein sequences (FASTA format)

4.  **Odorant-OR Pair Analysi**s: OdoriFy flexibility also supports the validation of OR-Odor pairs supplement by the user. Moreover, the explainable AI component of OdoriFy returns the sub-structure analysis (odor) and marking of key amino acids (OR) responsible for the predicted decision.
 <u>Input:</u> Chemical (SMILES) and Protein sequences (FASTA Format).

## Dependencies
1.  Python v.3.4.6 or higher
2.  TensorFlow v1.12
3.  rdkit v.2018.09.2
4. Conda Environment


## Installation Guidelines
 1. Install conda via: https://www.anaconda.com/products/individual
 2. Make the installer an executable file via: chmod u+x installer
 3. Run the installer via: ./installer

## Performance 


## Functionalities


## Citing
