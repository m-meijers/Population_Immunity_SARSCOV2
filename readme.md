## Population immunity predicts evolutionary trajectories of SARS-CoV-2

This repository contains code that is used for the analysis of population immunity against SARS-CoV-2. Continuously updated predictions on SARS-CoV-2 are shown on https://previr.app. 

This codebase consists of three sections:

1. Selection inference. The scripts in the directory selection_inference take genetic data from individual regions and infer time-dependent selection coefficients for each of the clade shifts: WT-Alpha, Alpha-Delta, Delta-BA.1, BA.1-BA.2, BA.2-BA.4/5, BA.4/5-BQ.1. Its outputs are stored in the folder "output". The input data is stored in the folder "DATA". We use genetic data with download date from GISAID on 2022-04-26 and 2023-04-01. 

2. Data processing and inference of antigenic parameters. The scripts Create_Data_ad_do.py, Create_Data_omicron.py, and Create_Immune_Trajectory_Data.py take data on inferred selection coefficients and combines the data with epidemiological data in the given countries. In these scripts, the population immunity is computed for all immune classes and variants covered in the paper. The script Create_Immune_Trajectory_Data.py preprocesses the combined data to a format that is read in for the scripts that produce figures. The script Linear_Regression_Final.py infers the antigenic parameters using the output data on each clade shift. The script Update_B.py infers antigenic parameters using data with a given data cut-off, which is used for predictions. The script Create_Average_Frequenc.py computes average frequency trajectories for all variants, used in plotting.

3. The scripts stored in Plot_Figures are used to produce each of the figures in the manuscript. It saves the pdf version of the figure in the same folder.