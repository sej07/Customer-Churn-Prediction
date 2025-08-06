## Customer Churn Prediction using Artificial Neural Networks

_This project aims to predict whether a customer will leave a bank using a simple feedforward Artificial Neural Network (ANN). The model uses 13 customer features and is trained on the Bank Turnover Dataset._

#### Dataset Details
- Dataset: Bank Turnover Dataset
- Source: Kaggle (https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling)
- Rows: 10000
- **Columns:**
  - `Credit Score` : Credit score of the customer
  - `Geography`: Country
  - `Gender`: Male/ Female
  - `Age`: Customer's Age
  - `Tenure`: Years with the bank
  - `Balance`: Account Balance
  - `NumOfProducts`: Number of bank products used
  - `HasCrCard`: Has credit card (0/1)
  - `IsActiveMember`: Active in the last year (0/1)
  - `EstimatedSalary`: Salary estimation
  - `Exited`: Target variable: Churn Variable

#### ML Workflow: 
1. Importing Libraries
    1. `numpy`, `pandas`
    2.  `matplotlib` and `seaborn` for visualization   
    3. `scikit-learn` for preprocessing and evaluation
    4. `tensorflow.keras` for ANN modeling
2. Data Loading
    1. Loaded dataset using `read_csv()`
3. Data Wrangling
    1. Encoded categorical variables (`Gender`, `Geography`)
    2. Removed identifier columns (`RowNumber`, `CustomerId`, `Surname`)  
    3. Checked for class imbalance and data consistency    
4. Model Architecture
    1. The model is a simple feedforward neural network built using the Keras Sequential API. It consists of:
        1. One input layer with 11 features
        2. One hidden layer containing 3 neurons with sigmoid activation
        3. One output layer with a single neuron using sigmoid activation to output churn probability (between 0 and 1)
    2. Loss Function: Binary Crossentropy
    3. Optimizer: Adam
    4. Activation Function: Sigmoid (for both hidden and output layers)
5.  Model Training
    1. Compiled and trained the model for 10 epochs. Training was done on the scaled feature set using binary crossentropy as the loss function.
6. Evaluation
    1. Metrics used: `Accuracy`

#### Results
Accuracy Score: 0.79

Loss: 0.43

#### Key Observation 
Customers with shorter tenure and low activity levels are more likely to churn.

#### Improvements
1. A deeper and complex architecture could capture more complex patterns
2. Loss can be decreased using regularization or dropout, resulting is reducing overfitting
3. More epochs and hyperparameter tuning could further improve accuracy

#### Visualizations
<img width="754" height="602" alt="Screenshot 2025-08-06 194515" src="https://github.com/user-attachments/assets/fc068b54-0223-417e-8969-7413fa738c62" />

#### Assumptions
1. All 13 features contribute meaningfully to the output

