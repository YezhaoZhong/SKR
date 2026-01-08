# SKR
Code for Smoothed Kernel Regression (SKR)




## .ipynb files and .py files
The .ipynb files are generating the results for Adverse Drug Reaction (ADR) profile predictions. We developed SKR and compared it with series of advanced methods. .py files contain functions for Nested Cross-Validation (CV) and CV. CV was used to tuned the hyperparameter for hold-out set. A toy data was designed to justify and clarify how SKR functions on the ADR data. We also learn how SKR affects the prediction of common ADRs (frequent ADRs) and rare ADRs (infrequent ADRs) and how the strength of smoother in SKR affects the performance. We also used this toy data to show the limitation of VKR, our previous study.

Required modules:
- pandas
- networkx
- numpy
- sklearn
- joblib
- warnings
- collections
- scipy
- seaborn
- matplotlib
- json
- itertools
- xml
- request
- rdkit


## Functions (.py files)

- [ADRprofileprediction.py](https://github.com/YezhaoZhong/SKR/blob/main/ADRprofilePrediction.py): This file contains the functions for Nested CV and CV of ADR profile prediction.
- [Model.py](https://github.com/YezhaoZhong/SKR/blob/main/Models.py): This file contains functions of loading hyperparemeters for tuning and prediction methods. It allows making profile prediction in the Nested CV and CV workflows of [ADRprofileprediction.py]. Also, it contains the code for used methods in this study, including SKR, Kernel Ridge Regression (KRR), Kernel Regression on V (VKR), Linear Neighbourhood Similarity Method (LNSM) with jaccard similarity or Regularized linear neighbour similarity (RLN), Support Vector Machine (SVM), Random Forest (RF), Boosted RF (BRF).




## Toy Data (.ipynb files)


### Input: 
- None

#### Code:
[toydata.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/toydata.ipynb): The smoother of SKR was adopted to a toy ADR data with common and rare ADR defined. The smoothed ADR data was visualized to show how the smoother works by heatmap.

#### Output: 
[/figs/](https://github.com/YezhaoZhong/SKR/tree/main/figs)
- [heatmapY.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapY.jpg): Heatmap of origin ADR toy data (Figure 2 A and Supplementary I Figure S 3 A).
- [heatmapYS.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapYS.jpg): Heatmap of the ADR toy data smoothed once (Figure 2 B).
- [heatmapYSS.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapYSS.jpg): Heatmap of the ADR toy data smoothed twice (Figure 2 C).
- [heatmapVKR.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapVKR.jpg): Heatmap of example prediction using VKR (Supplementary I Figure S 3 C).
- [heatmapKRR.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/heatmapKRR.jpg): Heatmap of example prediction using KR (Supplementary I Figure S 3 D).




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


### Code:
* Running Nested CV and CV on SIDER (or OFFSIDES) with all (or rare) ADR data used: [mainSIDER_all.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainSIDER_all.ipynb), [mainSIDER_rare.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainSIDER_rare.ipynb), [mainOFFSIDES_all.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainOFFSIDES_all.ipynb), [mainOFFSIDES_rare.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainOFFSIDES_rare.ipynb).
    * Output 1. [/results/](https://github.com/YezhaoZhong/SKR/tree/main/results) Formated results: A_results_B_C.xlsx (Table 1, Table 2, Table 3, Table 4, Supplementary D Table S 1, Table S 2, Table S 3 and Table S 4 and Supplementary E Table S 5, Table S 6, Table S 7 and Table S 8).
        * A: cv, nested_cv (cv is the results of hold-out set using hyperparameter tuned in CV).
        * B: SIDER, OFFSIDES.
        * C: all, rare.
        * For example, the results of Nested CV on predicting rare ADRs of SIDER: [nested_cv_results_SIDER_rare.xlsx](https://github.com/YezhaoZhong/SKR/blob/main/results/nested_cv_results_SIDER_rare.xlsx).
    * Output 2. [/results/](https://github.com/YezhaoZhong/SKR/tree/main/results) Raw results in dictionary: results_A_B.xml.
        * A: SIDER, OFFSIDES.
        * B: all, rare.
        * For example, the results of Nested CV on predicting rare ADRs of SIDER for all methods: [results_SIDER_rare.xml](https://github.com/YezhaoZhong/SKR/blob/main/results/results_SIDER_rare.xml)
    * Output 3. [hyperpars.xml](https://github.com/YezhaoZhong/SKR/blob/main/data/hyperpars.xml): Tuned hyperparameters for each method in Nested CV and CV, using different features. This file works as input when you want to skip the tuning step and used the tuned hyperparameters.
    * Output 4. [/results/](https://github.com/YezhaoZhong/SKR/tree/main/results) P-values of pairwise paired t-test for method comparison: pvalue_A_B_C_D.xlsx.
        * A: SIDER, OFFSIDES.
        * B: all, rare.
        * C: pathway, Chem, DGI, indication, target, transporter, enzyme.
        * D: AUPR, AUROC, AUPRperdrug, AUROCperdrug, AUPR+AUROC, AUPR+AUROCperdrug.
        * For example, the p-value of method comparison in rare ADRs of SIDER, using the pathway feature, and under the metric AUPR: [pvalue_SIDER_rare_pathway_AUPR.xlsx](https://github.com/YezhaoZhong/SKR/blob/main/results/pvalue_SIDER_rare_pathway_AUPR.xlsx)

* [SVM_RF.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/SVM_RF.ipynb): SVM, RF and BRF is not competitive as the others and time consuming. Therefore, we run them seperately to reduce the tuning time of [mainSIDER_all.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainSIDER_all.ipynb) and [mainOFFSIDES_all.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/mainOFFSIDES_all.ipynb).
    * Output 1. [hyperpars.xml](https://github.com/YezhaoZhong/SKR/blob/main/data/hyperpars.xml): Tuned hyperparameters for each method in Nested CV and CV, using different features (Table 1 and Table 2).

* [time.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/time.ipynb): This file compares the runtime of SKR, VKR, KRR, SVM, RF, and BRF on the hold-out set.




## Define Rare ADRs (.ipynb file)

### Input:
[/data/](https://github.com/YezhaoZhong/SKR/blob/main/data/)
* ADR data
    * [drug_se.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_se.tsv): Drug-ADR pairs from SIDER.
    * [OFFSIDES.csv](https://github.com/YezhaoZhong/SKR/blob/main/data/OFFSIDES.csv): Drug-ADR pairs from OFFSIDES.
* Feature data
    * [interactions.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/interactions.tsv): Drug-gene interactions from DGIdb.
    * [hyperpars.xml](https://github.com/YezhaoZhong/SKR/blob/main/data/hyperpars.xml): Tuned hyperparameters for each method in Nested CV and CV (hold-out set), using different features. This file works as input when you want to skip the tuning step and used the tuned hyperparameters.
 
### Code:
[define_rare.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/define_rare.ipynb): We drew the density plot of the ADR data. Then KRR was used to test how the noise ADRs, the rare ADRs and common ADRs affect the perdiction performance on SIDER.

### Output:
[/figs/](https://github.com/YezhaoZhong/SKR/blob/main/figs/)
* [SIDER_rare.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/SIDER_rare.jpg): The distribution of ADRs on SIDER, with frequencies < 50 highlighted (Figure 6 A).
* [OFFSIDES_rare.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/OFFSIDES_rare.jpg): The distribution of ADRs on OFFSIDES, with frequencies < 50 highlighted (Figure 6 B).
* [define_tau.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/define_tau.jpg): AUROC - tau curve of the naive method and KRR, where tau is the threshold of noise ADRs (Figure 3).


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


### Code:
We visualized nested CV results with boxplots using SIDER. Methods compared: SKR, VKR, and KRR. ADR frequency groups: <50 ([boxplot_0_50.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/boxplot_0_50.ipynb)), <100([boxplot_0_100.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/boxplot_0_100.ipynb)), <150 ([boxplot_0_150.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/boxplot_0_150.ipynb)), and >50 ([boxplot_50_all.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/boxplot_50_all.ipynb)).


### Output:
[/figs/](https://github.com/YezhaoZhong/SKR/blob/main/figs/)

Boxplots of the results of Nested CV with different metrics used: A_B.jpg (Figure 4, Supplementary A Figure S 1 and Supplementary B Figure S 2).
* A: AUPR, AUROC, AUPRperdrug, AUROCperdrug, AUPR+AUROC, AUPR+AUROCperdrug.
* B: 0_50, 0_100, 0_150, 50_all.
* For example, the boxplots of AUPR, evaluating the ADRs that has frequencies < 50: [AUPR_0_50.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/AUPR_0_50.jpg).





## Visualize Effect of Smoother (.ipynb files)


### Input:
[/data/](https://github.com/YezhaoZhong/SKR/blob/main/data/)
* ADR data
    * [drug_se.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_se.tsv): Drug-ADR pairs from SIDER.
* Feature data
    * [drug_indication.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_indication.tsv): Drug-indication pairs from SIDER.


### Code:
[smoother_function.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/smoother_function.ipynb): We studied how the strength the smoother affects the prediction performance, using the DGI feature and SIDER as ADR data.

### Output: 
[/figs/](https://github.com/YezhaoZhong/SKR/blob/main/figs/)

Metrics - smooth level (c) curves (Figure 5): 
* [AUPR-C.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/AUPR_C.jpg), [AUROC-C.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/AUROC_C.jpg), [AUPR+AUROC-C.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/AUPR+AUROC_C.jpg).
* In this case the curve of metric and per drug metric were drew in the same plot. For example, in [AUPR-C.jpg](https://github.com/YezhaoZhong/SKR/blob/main/figs/AUPR_C.jpg), curve of AUPR-c and AUPRperdrug are included.



## \*Data Preprocess and Update (.ipynb files)

This section only for updating data and generating data from raw data. If there is not requirement of using the latest data, please skip this section and use data in [/data/](https://github.com/YezhaoZhong/SKR/blob/main/data/).

The following data were downloaded directly from the website:
- SIDER 4.1 (including ADRs and indications) can be downloaded from: [http://sideeffects.embl.de/](http://sideeffects.embl.de/). 
- OFFSIDES is provided at: [https://tatonettilab.org/offsides/](https://tatonettilab.org/offsides/).
- DGIdb is available at: [https://www.dgidb.org/](https://www.dgidb.org/). 

And the following data need the provided code to generate:
Data from DrugBank including drug-related transporters, enzymes, and targets are extract from the DrugBank xml file downloaded from: [https://go.drugbank.com/releases/latest](https://go.drugbank.com/releases/5-1-12). 
The Chemical structure fingerprints were downloaded via API: https://pubchem.ncbi.nlm.nih.gov/.  
Drug-related pathways can be accessed via API from: https://www.genome.jp/kegg/. 

### Input: 
[/data/](https://github.com/YezhaoZhong/SKR/blob/main/data/)

* ADR data (Drug-ADR data was only used to generate the drug list used for API)
    * [drug_se.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_se.tsv): Drug-ADR pairs from SIDER.
    * [OFFSIDES.csv](https://github.com/YezhaoZhong/SKR/blob/main/data/OFFSIDES.csv): Drug-ADR pairs from OFFSIDES.

* DrugBank raw data
    * [full database.xml](https://go.drugbank.com/releases/5-1-12): xml file for DrugBank. This file contains all the information from DrugBank. Because of the size of the data, this file was not uploaded to github.

### Code:
[newData.ipynb](https://github.com/YezhaoZhong/SKR/blob/main/newData.ipynb): Code for generating drug-tranporter pairs, drug-enzyme pairs, drug-target pairs and chemical structure fingerprint.

### Output:
[/data/](https://github.com/YezhaoZhong/SKR/blob/main/data/)
* [drug_target.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_target.tsv): Drug-target pairs fetched from DrugBank.
* [drug_transporter.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_transporter.tsv): Drug-transporter pairs loaded from DrugBank.
* [drug_enzyme.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_enzyme.tsv): Drug-enzyme pairs fetched from DrugBank.
* [drug_chemsfp.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_chemsfp.tsv): Chemical structure fingerprint downloaded from PubChem.
* [drug_pathway.tsv](https://github.com/YezhaoZhong/SKR/blob/main/data/drug_pathway.tsv): Drug-pathway pairs from KEGG.

## Author
Yezhao Zhong, 
Cathal Seoighe, 
Haixuan Yang

## License

This project is licensed. See [LICENSE](https://github.com/YezhaoZhong/SKR/blob/main/LICENSE) file for details.

