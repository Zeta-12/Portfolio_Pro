# Machine Learning Projects

A collection of 13 machine learning projects built in Python, covering the full spectrum from classical algorithms to deep learning. Each project is self-contained with its own dataset, class-based implementation, and documented results.

## Tech stack

Python 3.12+ · scikit-learn · PyTorch · torchvision · XGBoost · NLTK · pandas · matplotlib · tqdm

## Projects

| # | Project | Algorithm(s) | Dataset |
|---|---------|-------------|---------|
| 01 | [Data Preprocessing](01_data_preprocessing/) | Imputation, encoding, scaling | 10 rows |
| 02 | [Salary Prediction](02_salary_prediction/) | Simple Linear Regression | 30 rows |
| 03 | [Startup Profit Prediction](03_startup_profit_prediction/) | Multiple Linear Regression | 50 startups |
| 04 | [Position Salary Benchmark](04_position_salary_regression/) | Polynomial, SVR, Decision Tree, Random Forest | 10 rows |
| 05 | [Ad Purchase Prediction](05_ad_purchase_prediction/) | 7 classifiers + GridSearchCV | 400 users |
| 06 | [Mall Customer Segmentation](06_mall_customer_segmentation/) | K-Means, Hierarchical Clustering | 200 customers |
| 07 | [Market Basket Analysis](07_market_basket_analysis/) | Apriori, ECLAT | 7,500 transactions |
| 08 | [Ad CTR Optimization](08_ad_ctr_optimization/) | UCB, Thompson Sampling | 10,000 rounds |
| 09 | [Restaurant Sentiment Analysis](09_restaurant_sentiment_analysis/) | Naive Bayes + Bag of Words | 1,000 reviews |
| 10 | [Bank Churn Prediction](10_bank_churn_prediction/) | ANN (PyTorch) | 10,000 customers |
| 11 | [Cat vs Dog Classification](11_cat_dog_image_classification/) | CNN (PyTorch) | Oxford-IIIT Pet |
| 12 | [Wine Quality Classification](12_wine_quality_classification/) | PCA, LDA, Kernel PCA + Logistic Regression | 178 wines |
| 13 | [XGBoost Tabular Classifier](13_xgboost_tabular/) | XGBoost + 10-fold CV | Tabular dataset |

## Setup

```bash
git clone <repo-url>
cd project

python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Running a project

```bash
cd 02_salary_prediction
python simple_linear_regression.py
```

Most projects complete in under a minute. Projects 05 and 10–11 take a few minutes due to cross-validation or neural network training.

Training hyperparameters for projects 10 and 11 can be overridden via CLI:

```bash
python artificial_neural_network.py --epochs 50 --lr 0.0005 --save-model my_model.pt
python convolutional_neural_network.py --epochs 10 --batch-size 64 --weights-path weights/resnet18-f37072fd.pth
```

## Repository structure

```
project/
├── 01_data_preprocessing/
│   ├── data_preprocessing_tools.py
│   ├── Data.csv
│   └── README.md
├── 02_salary_prediction/
│   ├── simple_linear_regression.py
│   ├── Salary_Data.csv
│   └── README.md
├── ...
├── requirements.txt
└── README.md
```

Each project folder contains the Python source file, the dataset it uses, and a README with expected results and sample output plots.
