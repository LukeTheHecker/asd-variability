# :rotating_light: This repository is under construction and will be complete upon publication of the manuscript :rotating_light:

---

# Code basis for: "Altered EEG Variability on Different Time Scales in Participants with Autism Spectrum Disorder - An Exploratory Study"
This repository showcases the code used for the analyses presented in the paper **"Altered EEG Variability on Different Time Scales in Participants with Autism Spectrum Disorder - An Exploratory Study"** which is currently under review in *Nature Scientific Reports*. 

The data used for in this work can not be shared.

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
Furthermore, this notebook also produces Figures and Tables.