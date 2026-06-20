# Customer Segmentation and Delivery Policy Recommendation

## Project Overview

This project analyzes wholesale customer purchasing behavior to identify hidden customer segments using unsupervised machine learning techniques. The discovered segments are then used to build a supervised classification model capable of assigning new customers to the appropriate segment and recommending a suitable delivery policy.

The project follows an end-to-end machine learning workflow including data preprocessing, dimensionality reduction, clustering, customer profiling, and predictive modeling.

---

## Business Problem

A wholesale distributor serves multiple types of customers, including:

- Hotels
- Restaurants
- Cafes (HoReCa)
- Retail Stores
- Supermarkets

The company is considering reducing delivery frequency from 5 days per week to 3 days per week for customers who can tolerate less frequent deliveries.

Since customer types are not explicitly labeled, the goal is to:

1. Discover customer segments based on purchasing behavior.
2. Understand the characteristics of each segment.
3. Predict the segment of future customers.
4. Recommend the most appropriate delivery schedule.

---

## Dataset Features

The dataset contains annual spending in six product categories:

| Feature | Description |
|----------|-------------|
| Fresh | Fresh products |
| Milk | Dairy products |
| Grocery | Grocery products |
| Frozen | Frozen products |
| Detergents_Paper | Cleaning and paper products |
| Delicatessen | Delicatessen products |

---

## Project Workflow

### 1. Data Preprocessing

- Removed irrelevant features (`Region`, `Channel`)
- Detected and removed outliers
- Applied logarithmic transformation to reduce skewness

```text
Original Data
      ↓
Log Transformation
      ↓
Outlier Removal
      ↓
Clean Dataset
```

---

### 2. Principal Component Analysis (PCA)

PCA was used to reduce dimensionality while preserving most of the variance.

#### Explained Variance

| Principal Component | Variance Explained |
|--------------------|-------------------|
| PC1 | 44.30% |
| PC2 | 26.38% |
| PC3 | 12.31% |
| PC4 | 10.12% |
| PC5 | 4.85% |
| PC6 | 2.04% |

#### Cumulative Variance

| Components | Variance Retained |
|------------|------------------|
| PC1 + PC2 | 70.68% |
| PC1 + PC2 + PC3 + PC4 | 93.61% |

Two principal components were used for visualization purposes.

```text
6 Features
      ↓
PCA
      ↓
2-Dimensional Representation
```

---

### 3. Customer Segmentation

Two clustering approaches were considered:

- K-Means
- Gaussian Mixture Models (GMM)

#### Why GMM?

Customer behavior showed overlap between segments.

K-Means performs hard clustering:

```text
Customer → One Cluster Only
```

GMM performs soft clustering:

```text
Customer → Probability Distribution Across Clusters
```

Example:

```text
Retail      = 70%
HoReCa      = 30%
```

This makes GMM more suitable for customers exhibiting mixed purchasing behavior.

---

### 4. Cluster Evaluation

The Silhouette Score was used to evaluate clustering quality.

#### Silhouette Scores

| Number of Clusters | Score |
|-------------------|--------|
| 2 | 0.422 |
| 3 | 0.404 |
| 4 | 0.293 |
| 5 | 0.300 |
| 6 | 0.326 |
| 7 | 0.324 |
| 8 | 0.296 |
| 9 | 0.307 |
| 10 | 0.310 |

The best score was achieved with:

```text
2 Clusters
```

---

## Segment Interpretation

Cluster centers were transformed back into the original business scale using:

```text
PCA Inverse Transform
      ↓
Exponential Transform
      ↓
Original Spending Values
```

### Segment 0

| Feature | Value |
|----------|--------:|
| Fresh | 8953 |
| Milk | 2114 |
| Grocery | 2765 |
| Frozen | 2075 |
| Detergents_Paper | 353 |
| Delicatessen | 732 |

#### Characteristics

- High Fresh spending
- High Frozen spending
- Low Grocery spending
- Low Detergents spending

#### Business Interpretation

This segment likely represents:

- Hotels
- Restaurants
- Cafes
- Catering Services

Classification:

```text
HoReCa Segment
```

---

### Segment 1

| Feature | Value |
|----------|--------:|
| Fresh | 3552 |
| Milk | 7837 |
| Grocery | 12219 |
| Frozen | 870 |
| Detergents_Paper | 4696 |
| Delicatessen | 962 |

#### Characteristics

- High Grocery spending
- High Milk spending
- High Detergents spending
- Lower Fresh spending

#### Business Interpretation

This segment likely represents:

- Supermarkets
- Grocery Stores
- Retail Chains
- Convenience Stores

Classification:

```text
Retail Segment
```

---

## Validation Against Hidden Labels

The dataset originally contained a hidden feature:

```text
Channel
```

with two categories:

- HoReCa
- Retail

This feature was intentionally excluded during clustering.

After reintroducing it, the discovered clusters showed strong alignment with the actual business categories, validating that customer spending behavior contains enough information to distinguish between HoReCa and Retail customers.

---

## Building a Supervised Learning Model

The cluster assignments generated by GMM were used as labels.

```text
Spending Features
        ↓
GMM Clustering
        ↓
Segment Labels
        ↓
Supervised Learning
```

### Input Features

```python
[
    Fresh,
    Milk,
    Grocery,
    Frozen,
    Detergents_Paper,
    Delicatessen
]
```

### Target Variable

```python
Customer Segment
```

---

## Models Evaluated

### Logistic Regression

F1 Score:

```text
0.9836
```

### Random Forest

F1 Score:

```text
0.9677
```

### XGBoost

F1 Score:

```text
0.9508
```

### Best Model

```text
Logistic Regression
```

The high performance indicates that the customer segments discovered through clustering are highly predictable from spending behavior.

---

## Deployment Workflow

For a new customer:

```text
Customer Spending
        ↓
Log Transformation
        ↓
Logistic Regression Model
        ↓
Predicted Segment
        ↓
Delivery Recommendation
```

### Example

Input:

```text
Fresh = 3000
Milk = 9000
Grocery = 15000
Frozen = 800
Detergents_Paper = 5000
Delicatessen = 1000
```

Prediction:

```text
Retail Segment
```

Recommendation:

```text
3 Delivery Days Per Week
```

---

## Business Recommendations

### HoReCa Customers

Characteristics:

- Depend heavily on Fresh products
- Require frequent replenishment
- Sensitive to delivery delays

Recommended Service:

```text
5 Deliveries Per Week
```

---

### Retail Customers

Characteristics:

- Purchase products with longer shelf life
- Store larger inventories
- Less dependent on frequent deliveries

Recommended Service:

```text
3 Deliveries Per Week
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- PCA
- Gaussian Mixture Models (GMM)
- K-Means
- Logistic Regression
- Random Forest
- XGBoost

---

## Key Outcomes

- Reduced six-dimensional purchasing data into interpretable principal components.
- Identified meaningful customer segments using unsupervised learning.
- Validated discovered segments against real business categories.
- Built a highly accurate supervised classifier to predict customer segments.
- Developed a framework for assigning delivery policies to future customers.

---

## Project Pipeline

```text
Original Data (6 Features)
        ↓
Log Transformation
        ↓
Outlier Removal
        ↓
PCA (6D → 2D)
        ↓
Gaussian Mixture Model (GMM)
        ↓
Customer Segmentation
        ↓
Cluster Interpretation
        ↓
Generate Segment Labels
        ↓
Train Logistic Regression
        ↓
Predict New Customer Segment
        ↓
Recommend Delivery Policy
```