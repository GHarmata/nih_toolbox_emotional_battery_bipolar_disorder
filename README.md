# Modified Machine Learning Code Used in Gritter & Harmata et al. (2025)

This repository contains the code used to conduct the machine learning random forest 
analysis with 5-fold nested cross-validation described in the paper "Associations between 
NIH Toolbox Emotion Battery measures and previous suicide attempt in bipolar I disorder" 
by Gritters & Harmata et al. (2025), PMID: 39672472.  The manuscript is available at 
https://www.sciencedirect.com/science/article/pii/S0165032724020251?via%3Dihub#t0015

## Purpose

This code was created to determine whether the NIH Toolbox Emotional Battery measures
(along with age and self-reported sex) would be able to classify whether a participant 
with bipolar disorder type I had a past suicide attempt, and which variables are the most
important.

## Understanding the Repository

The primary analysis file is "BD_nih_toolbox_machine_learning.R".  The code relies on the 
following packages: *nestedcv* (Lewis et al., 2023), caret (Kuhn, 2008), 
*randomForest* (Liaw and Wiener, 2002), *randomForestExplainer* (Paluszynska et al., 
2020), the *tidyverse* (Wickham et al., 2019), *ggprism* (Dawson, 2022), *ggrepel* 
(Slowikowski, 2023), and *DescTools* (Signorell, 2024).  This code also relies on two 
additional files, `"updated_functions_nih_toolbox_project.R"` and `"caret_custom_rf_v2.R".`

The file `"updated_functions_nih_toolbox_project.R"` overwrites several functions from the 
imported packages.  This was done to allow *nestedcv* and *caret* to work with 
*randomForestExplainer*, and to remove the automatic scaling of feature 
importance. Similarly, the file `"caret_custom_rf_v2.R"` allows *caret* to work with 
*randomForestExplainer* to determine variable importance.  


## Acknowledgements

This code would not be possible without the packages it is built upon.  We thank the 
creators of those packages for their excellent work.  We also thank our study participants,
without which our findings would not exist.

The data used by this code was collected at the University of Iowa and relied on funding
from the NIMH (R01MH111578, R01MH125838), NCATS (UL1TR002537), the Roy J. Carver 
Charitable Trust, and the Iowa Neuroscience Institute.  Additional support to authors was 
provided by an NIH training grant (T32MH019113) and the U.S. Department of Veterans 
Affairs.

## Contact

Please contact gail-harmata@uiowa.edu with questions about this code.
