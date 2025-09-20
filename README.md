# SKR
Code for Smoothed Kernel Regression (SKR)



# .ipynb files and .py files
The .ipynb files are generating the results for Adverse Drug Reaction (ADR) profile predictions. We developed SKR and compared it with series of advanced methods. .py files contain functions for Nested Cross-Validation (CV) and CV. A toy data was designed to justify and clarify how SKR functions on the ADR data. We also learn how SKR affects the prediction of common ADRs and rare ADRs and how the strength of smoother in SKR affects the performance. 

## Toy data
- [toydata.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/toydata.ipynb): The smoother of SKR was adopted to a toy ADR data with common and rare ADR defined. The smoothed ADR data was visualized to show how the smoother works by heatmap.

Input: 
- None

Output: [/figs/](https://github.com/YezhaoZhong/SKR/tree/main/figs)
- [heatmapY.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapY.jpg): Heatmap of origin ADR toy data.
- [heatmapYS.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapYS.jpg): Heatmap of the ADR toy data smoothed once.
- [heatmapYSS.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapYSS.jpg): Heatmap of the ADR toy data smoothed twice.

## Functions (.py files)

- [ADRprofileprediction.py](https://github.com/YezhaoZhong/SKR/blob/main/ADRprofilePrediction.py): This file contains the functions for Nested CV and CV of ADR profile prediction.
- [Model.py](https://github.com/YezhaoZhong/SKR/blob/main/Models.py): This file contains functions of loading hyperparemeters for tuning and prediction methods. It allows making profile prediction in the Nested CV and CV workflows of [ADRprofileprediction.py]. Also, it contains the code for used methods in this study, including SKR, Kernel Ridge Regression (KRR), Kernel Regression on V (VKR), Linear Neighbourhood Similarity Method (LNSM) with jaccard similarity or Regularized linear neighbour similarity (RLN), Support Vector Machine (SVM), Random Forest (RF), Boosted RF (BRF).

## ADR Profile prediction for all ADRs and rare ADRs (.ipynb files)

- [mainSIDER_all.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainSIDER_all.ipynb): Running Nested CV and CV on SIDER with all ADR data used.
- [mainSIDER_rare.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainSIDER_rare.ipynb): Running Nested CV and CV on SIDER with rare ADR data used.
- [mainOFFSIDES_all.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainOFFSIDES_all.ipynb): Running Nested CV and CV on OFFSIDES with all ADR data used.
- [mainOFFSIDES_rare.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainOFFSIDES_rare.ipynb): Running Nested CV and CV on OFFSIDES with rare ADR data used.

Input:[/data/](https://github.com/YezhaoZhong/SKR/blob/main/data/)

ADR data
- [drug_se.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_se.tsv): Drug-ADR pairs from SIDER.
- [OFFSIDES.csv](https://github.com/YezhaoZhong/SKR/blob/main/data/OFFSIDES.csv): Drug-ADR pairs from OFFSIDES.

Feature data
- [drug_target.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_target.tsv): Drug-target pairs fetched from DrugBank.
- [drug_transporter.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_transporter.tsv): Drug-transporter pairs loaded from DrugBank.
- [drug_enzyme.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_enzyme.tsv): Drug-enzyme pairs fetched from DrugBank.
- [drug_chemsfp.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_chemsfp.tsv): Chemical structure fingerprint downloaded from PubChem.
- [interactions.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/interactions.tsv): Drug-gene interactions from DGIdb.
- [drug_pathway.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_pathway.tsv): Drug-pathway pairs from KEGG.
- [drug_indication.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_indication.tsv): Drug-indication pairs from SIDER.
\* [hyperpars.xml](https://github.com/YezhaoZhong/SKR/blob/main/data/hyperpars.xml): Tuned hyperparameters for each method in Nested CV and CV, using different features. This file works as input when you want to skip the tuning step and used the tuned hyperparameters.

- [SVM_RF.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/SVM_RF.ipynb): SVM, RF and BRF is not competitive as the others and time consuming. Therefore, we run them seperately to reduce the tuning time of [mainSIDER_all.ipynb] and [mainOFFSIDES_all.ipynb].

- []
