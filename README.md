# Student Depression Classification & Interactive Model Evaluation

## a. Problem Statement
Mental health issues among university students present a critical challenge impacting academic performance and personal well-being. This project aims to build an end-to-end Machine Learning classifier to predict student depression status based on socio-demographic, academic, lifestyle, and health indicators.

## b. Dataset Description
- **Dataset:** Student Depression Dataset
- **Total Instances:** 4,464 records
- **Total Features:** 12 predictive features (`Gender`, `Age`, `Family_background`, `Relationship_status`, `Marital_status`, `Living_style`, `Current_semester`, `CGPA`, `friend_circle`, `Health_condition`, `Cost_without_study`, `Home_division`)
- **Target Variable:** `Depression` (`Yes` / `No`)

## c. GitHub Repository Link
[Insert Your GitHub Repository URL Here]

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8555 | 0.9023 | 0.7681 | 0.6625 | 0.7114 | 0.6188 |
| Decision Tree | 0.9922 | 0.9953 | 0.9874 | 0.9833 | 0.9854 | 0.9800 |
| KNN | 0.8903 | 0.9483 | 0.8480 | 0.7208 | 0.7793 | 0.7110 |
| Naive Bayes | 0.8835 | 0.8961 | 0.8119 | 0.7375 | 0.7729 | 0.6963 |
| Support Vector Machine | 0.9026 | 0.9604 | 0.8844 | 0.7333 | 0.8018 | 0.7437 |

### Model Observations
- **Logistic Regression:** Provides a strong linear baseline with an Accuracy of 85.55% and AUC of 0.9023, though recall for the positive class (0.6625) is relatively lower than tree-based approaches.
- **Decision Tree:** Outperforms all models across every evaluation metric, achieving an Accuracy of 99.22%, F1-score of 0.9854, and MCC of 0.9800, capturing subtle feature interactions effectively.
- **KNN:** Achieves an Accuracy of 89.03% and AUC of 0.9483, benefiting from clean feature normalization across neighbor distance calculations.
- **Naive Bayes:** Shows solid performance (Accuracy 88.35%), maintaining balanced precision and recall despite its feature independence assumption.
- **Support Vector Machine:** Demonstrates high precision (88.44%) and AUC (0.9604) with an overall Accuracy of 90.26%.

### Overall Winner
**Decision Tree Classifier** is the best performing model for this dataset, achieving top metrics across Accuracy (99.22%), AUC (0.9953), and MCC (0.9800).