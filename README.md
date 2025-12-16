# Steel Plates Faults Classification 

Machine Learning project for classifying surface defects in steel plates using classical ML algorithms.

## Project Overview
This project uses the **Steel Plates Faults Dataset** to classify manufacturing defects into 7 fault categories using supervised machine learning.

The original dataset is **multi-label**, but in this project it was converted to a **multi-class classification** problem by selecting the dominant fault per sample.

## Dataset
- Source: UCI Machine Learning Repository
- Samples: 1941
- Features: 27 numerical features
- Fault Classes (7):
  - Pastry
  - Z_Scratch
  - K_Scratch
  - Stains
  - Dirtiness
  - Bumps
  - Other_Faults

## Machine Learning Pipeline
1. Load `.NNA` dataset
2. Feature & label separation
3. Multi-label → Multi-class conversion
4. Train-test split (80/20)
5. Model training:
   - Random Forest (with GridSearchCV)
6. Model evaluation:
   - Accuracy
   - Classification report
7. Feature importance analysis
8. Model comparison:
   - Random Forest
   - Logistic Regression
   - SVM
   - KNN

##  Results
- Best model: **Random Forest**
- Accuracy: ~90% (may vary depending on random state)

## Visualizations
- Fault distribution in test set
- Top 10 most important features

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib

## Authur
- Mohamedin Khalafalla
- Mechanical Engineering & Machine Learning 
