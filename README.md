# Heart Disease Prediction
Comparisons of different machine learning algorithms predicting whether someone has heart disease from 14 biological attributes.

The maximum accuracy is <b><i>91.8%</i></b>, achieved using the K-Nearest Neighbors classification algorithm with eight neighbors.

## Dataset

<code>heart.csv</code>, collected from the [Kaggle Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci) challenge, contains 14 biological attributes of 303 people, including whether the person has heart disease or not.

### Biological Attributes

Features used for training machine learning models on, including the special binary class label <b><i>target</b></i>, describing whether heart disease was detected.

1. <b><i>age</i></b>: Age in years
2. <b><i>ca</i></b>: Number of major blood vessels (0-3)
3. <b><i>chol</i></b>: Serum cholestrol in mg/dl
4. <b><i>cp</i></b>: Chest pain type
    * Value 1: Typical angina
    * Value 2: Atypical angina
    * Value 3: Non-anginal pain
    * Value 4: Asymptomatic
5. <b><i>exang</i></b>: Exercise induced angina (1 = yes; 0 = no)
6. <b><i>fbs</i></b>: fasting blood sugar > 120 mg/dl (1 = true; 0 = no)
7. <b><i>oldpeak</i></b>: ST depression induced by exercise relative to rest
8. <b><i>restecg</i></b>: Resting electrocardiographic results
    * Value 0: Normal
    * Value 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    * Value 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria
9. <b><i>sex</i></b>: Sex (1 = male; 0 = female)
10. <b><i>slope</i></b>: The slope of the peak exercise ST segment
    * Value 1: Upsloping
    * Value 2: Flat
    * Value 3: Downsloping
11. <b><i>target</i></b>: Heart disease detection (0 = disease; 1 = no disease)
12. <b><i>thal</i></b>: Thalium stress test
    * Value 3: normal
    * Value 6: fixed defect
    * Value 7: reversibe defect
13. <b><i>thalach</i></b>: Maximum heart rate achieved in bpm
14. <b><i>trestbps</i></b>: Resting blood pressure (in mmHg on admission to the hospital)

## Data Preprocessing

The following data preprocessing methods are used:

* Convert categorical column names to indicator variables using <code>pandas.get_dummies(dataset)</code>.
* Scale some columns with scalable columns, such as 'age', 'chol', 'oldpeak', 'thalach', 'trestbps', 'chol'.
* Split dataset into 80% training and 20% testing.

A correlation matrix and historgrams of the distributions of attributes are also generated.

## Machine Learning Algorithms

The machine learning algorithms with their accuracies are:

* K-Nearest Neighbors; <b><i>91.8%</i></b> with 8 neighbors
* Support Vector Classifier; <b><i>90.16%</i></b> with rbf activation
* Decision Tree Classifier, <b><i>85.25%</i></b> with 28 maximum features


Plots of all algorithms with accuracies are generated using <code>matplotlib.pyplot</code>.

## Dependencies

* Python 3.6+
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

## Resources

* Kaggle Heart Disease UCI &mdash;  
    https://www.kaggle.com/ronitf/heart-disease-uci
* UCI Machine Learning Repository: Heart Disease Data Set &mdash;  
    https://archive.ics.uci.edu/ml/datasets/Heart+Disease
