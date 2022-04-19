# Code basis Repository:
## "Altered EEG Variability on Different Time Scales in Participants with Autism Spectrum Disorder - An Exploratory Study"
This repository showcases the code used for the analyses presented in the paper **"Altered EEG Variability on Different Time Scales in Participants with Autism Spectrum Disorder - An Exploratory Study"** which is currently under review. 

The data itself can not be shared, unfortunately. 

## Get started

0. Setup  
A full list of python packages used for the analyses can be found [here](https://github.com/LukeTheHecker/asd-variability/blob/main/requirements.txt).  
You can install it using pip:
```
pip install -r requirements.txt
```

1. Preprocess Data  
The preprocessing of raw EEG files to clean epochs is done in [notebooks/1_preprocessing.ipynb](https://github.com/LukeTheHecker/asd-variability/blob/main/notebooks/1_preprocessing.ipynb).

2. Analysis of Variability  
The calculation of all variability metrics presented in the paper is done in [notebooks/2_data_analysis.ipynb](https://github.com/LukeTheHecker/asd-variability/blob/main/notebooks/2_data_analysis.ipynb).
Furthermore, this notebook also produces all Figures found in the results section and produces tables with the statistics.

3. Classification and Correlations
This notebook creates Table 1 and Table 2 in the manuscript: [notebooks/3_classification_and_distances.ipynb](https://github.com/LukeTheHecker/asd-variability/blob/main/notebooks/3_classification_and_distances.ipynb).