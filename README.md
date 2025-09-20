# SKR
Code for Smoothed Kernel Regression (SKR)




## .ipynb files and .py files
The .ipynb files are generating the results for Adverse Drug Reaction (ADR) profile predictions. We developed SKR and compared it with series of advanced methods. .py files contain functions for Nested Cross-Validation (CV) and CV. A toy data was designed to justify and clarify how SKR functions on the ADR data. We also learn how SKR affects the prediction of common ADRs and rare ADRs and how the strength of smoother in SKR affects the performance. 




## Functions (.py files)

- [ADRprofileprediction.py](https://github.com/YezhaoZhong/SKR/blob/main/ADRprofilePrediction.py): This file contains the functions for Nested CV and CV of ADR profile prediction.
- [Model.py](https://github.com/YezhaoZhong/SKR/blob/main/Models.py): This file contains functions of loading hyperparemeters for tuning and prediction methods. It allows making profile prediction in the Nested CV and CV workflows of [ADRprofileprediction.py]. Also, it contains the code for used methods in this study, including SKR, Kernel Ridge Regression (KRR), Kernel Regression on V (VKR), Linear Neighbourhood Similarity Method (LNSM) with jaccard similarity or Regularized linear neighbour similarity (RLN), Support Vector Machine (SVM), Random Forest (RF), Boosted RF (BRF).




## ADR Profile Prediction for All ADRs and Rare ADRs (.ipynb files)


### Input:
[/data/](https://github.com/YezhaoZhong/SKR/blob/main/data/)
* ADR data
    * [drug_se.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_se.tsv): Drug-ADR pairs from SIDER.
    * [OFFSIDES.csv](https://github.com/YezhaoZhong/SKR/blob/main/data/OFFSIDES.csv): Drug-ADR pairs from OFFSIDES.
* Feature data
    * [drug_target.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_target.tsv): Drug-target pairs fetched from DrugBank.
    * [drug_transporter.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_transporter.tsv): Drug-transporter pairs loaded from DrugBank.
    * [drug_enzyme.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_enzyme.tsv): Drug-enzyme pairs fetched from DrugBank.
    * [drug_chemsfp.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_chemsfp.tsv): Chemical structure fingerprint downloaded from PubChem.
    * [interactions.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/interactions.tsv): Drug-gene interactions from DGIdb.
    * [drug_pathway.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_pathway.tsv): Drug-pathway pairs from KEGG.
    * [drug_indication.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_indication.tsv): Drug-indication pairs from SIDER.
    * [hyperpars.xml](https://github.com/YezhaoZhong/SKR/blob/main/data/hyperpars.xml): Tuned hyperparameters for each method in Nested CV and CV (hold-out set), using different features. This file works as input when you want to skip the tuning step and used the tuned hyperparameters.


### Code

* Running Nested CV and CV (hold-out set) on SIDER (or OFFSIDES) with all (or rare) ADR data used: [mainSIDER_all.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainSIDER_all.ipynb), [mainSIDER_rare.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainSIDER_rare.ipynb), [mainOFFSIDES_all.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainOFFSIDES_all.ipynb), [mainOFFSIDES_rare.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainOFFSIDES_rare.ipynb).
    * Output 1. [/results/](https://github.com/YezhaoZhong/SKR/tree/main/results) Formated results: A_results_B_C.xlsx.
        * A: cv, nested_cv.
        * B: SIDER, OFFSIDES.
        * C: all, rare.
        * For example, the results of Nested CV on predicting rare ADRs of SIDER: [nested_cv_results_SIDER_rare.xlsx](https://github.com/YezhaoZhong/SKR/blob/main/results/nested_cv_results_SIDER_rare.xlsx)
    * Output 2. [/results/](https://github.com/YezhaoZhong/SKR/tree/main/results) Raw results in dictionary: results_A_B.xml.
        * A: SIDER, OFFSIDES.
        * B: all, rare.
        * For example, the results of Nested CV on predicting rare ADRs of SIDER for all methods: [results_SIDER_rare.xml](https://github.com/YezhaoZhong/SKR/blob/main/results/results_SIDER_rare.xml)
    * Output 3. [hyperpars.xml](https://github.com/YezhaoZhong/SKR/blob/main/data/hyperpars.xml): Tuned hyperparameters for each method in Nested CV and CV, using different features. This file works as input when you want to skip the tuning step and used the tuned hyperparameters.
    * Output 4. [/results/](https://github.com/YezhaoZhong/SKR/tree/main/results) P-values of method comparison: pvalue_A_B_C_D.xlsx.
    * A: SIDER, OFFSIDES.
    * B: all, rare.
    * C: pathway, Chem, DGI, indication, target, transporter, enzyme.
    * D: AUPR, AUROC, AUPRperdrug, AUROCperdrug, AUPR+AUROC, AUPR+AUROCperdrug.

* [SVM_RF.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/SVM_RF.ipynb): SVM, RF and BRF is not competitive as the others and time consuming. Therefore, we run them seperately to reduce the tuning time of [mainSIDER_all.ipynb] and [mainOFFSIDES_all.ipynb].
    * Output 1. [hyperpars.xml](https://github.com/YezhaoZhong/SKR/blob/main/data/hyperpars.xml): Tuned hyperparameters for each method in Nested CV and CV, using different features.

* [time.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/time.ipynb):This file compares the runtime of SKR, VKR, KRR, SVM, RF, and BRF on the hold-out set.




## Toy Data (.ipynb files)


### Input: 
- None

#### Code
[toydata.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/toydata.ipynb): The smoother of SKR was adopted to a toy ADR data with common and rare ADR defined. The smoothed ADR data was visualized to show how the smoother works by heatmap.

#### Output: 
[/figs/](https://github.com/YezhaoZhong/SKR/tree/main/figs)
- [heatmapY.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapY.jpg): Heatmap of origin ADR toy data.
- [heatmapYS.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapYS.jpg): Heatmap of the ADR toy data smoothed once.
- [heatmapYSS.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapYSS.jpg): Heatmap of the ADR toy data smoothed twice.




## Visualize Effect of Drug Frequencies (.ipynb files)


### Input:
[/data/](https://github.com/YezhaoZhong/SKR/blob/main/data/)
* ADR data
    * [drug_se.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_se.tsv): Drug-ADR pairs from SIDER.
* Feature data
    * [drug_target.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_target.tsv): Drug-target pairs fetched from DrugBank.
    * [drug_transporter.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_transporter.tsv): Drug-transporter pairs loaded from DrugBank.
    * [drug_enzyme.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_enzyme.tsv): Drug-enzyme pairs fetched from DrugBank.
    * [drug_chemsfp.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_chemsfp.tsv): Chemical structure fingerprint downloaded from PubChem.
    * [interactions.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/interactions.tsv): Drug-gene interactions from DGIdb.
    * [drug_pathway.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_pathway.tsv): Drug-pathway pairs from KEGG.
    * [drug_indication.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_indication.tsv): Drug-indication pairs from SIDER.
    * [hyperpars.xml](https://github.com/YezhaoZhong/SKR/blob/main/data/hyperpars.xml): Tuned hyperparameters for each method in Nested CV and CV, using different features. 


### Code


## Visualize Effect of Smoother
