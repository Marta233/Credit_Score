# Credit Scoring Model

## Project Overview

This project focuses on building a **Credit Scoring Model** for a buy-now-pay-later service using a dataset containing transactional data. The primary goal is to predict customer creditworthiness based on transactional and categorical data. The dataset contains various fields such as `TransactionId`, `CustomerId`, `ProductCategory`, `ChannelId`, and more. Categorical encoding (label encoding and one-hot encoding) is applied to convert categorical features into a format suitable for machine learning algorithms.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
  - [Label Encoding](#label-encoding)
  - [One-Hot Encoding](#one-hot-encoding)
  - [Counting Value Occurrences](#counting-value-occurrences)
- [Usage](#usage)
  - [Dependencies](#dependencies)
  - [Running the Project](#running-the-project)
- [Methods](#methods)
  - [Label Encoding](#label-encoding)
  - [One-Hot Encoding](#one-hot-encoding)
  - [Counting Value Occurrences](#counting-value-occurrences)
- [Model Training](#model-training)
- [Contributions](#contributions)
- [License](#license)

---

## Dataset

The dataset used in this project contains transactional data for customers making purchases via an eCommerce platform. Key columns include:

- `TransactionId`: Unique identifier for each transaction.
- `BatchId`: Identifier for batch processing.
- `AccountId`: Customer account identifier.
- `SubscriptionId`: Subscription information related to the account.
- `CustomerId`: Unique identifier for customers.
- `CurrencyCode`: The currency used for the transaction.
- `CountryCode`: The country of the transaction.
- `ProviderId`: Identifier for the service provider.
- `ProductId`: Unique identifier for the product.
- `ProductCategory`: Type of product purchased.
- `ChannelId`: The channel through which the transaction occurred.
- `Amount`: The amount involved in the transaction.
- `Value`: The value of the transaction.
- `TransactionStartTime`: Timestamp of the transaction.
- `PricingStrategy`: The pricing strategy used for the transaction.
- `FraudResult`: Whether the transaction was flagged as fraudulent (0 = No, 1 = Yes).

### Example Data

| TransactionId      | BatchId      | AccountId | SubscriptionId | CustomerId | CurrencyCode | CountryCode | ProviderId  | ProductId | ProductCategory     | ChannelId   | Amount | Value | TransactionStartTime     | PricingStrategy | FraudResult |
|--------------------|--------------|-----------|----------------|------------|---------------|-------------|-------------|-----------|----------------------|-------------|--------|-------|--------------------------|-----------------|-------------|
| TransactionId_11879| BatchId_61924| AccountId_661 | SubscriptionId_4753 | CustomerId_1001 | UGX           | 256         | ProviderId_5| ProductId_15| airtime              | ChannelId_3 | 10000  | 10000 | 2018-11-16T07:53:19Z   | 2               | 0           |
| TransactionId_36212| BatchId_23377| AccountId_661 | SubscriptionId_4753 | CustomerId_1001 | UGX           | 256         | ProviderId_6| ProductId_10| financial_services    | ChannelId_3 | 10000  | 10000 | 2018-11-16T08:16:24Z   | 2               | 0           |
| TransactionId_64687| BatchId_30141| AccountId_4841 | SubscriptionId_3829 | CustomerId_1001 | UGX           | 256         | ProviderId_4| ProductId_6| financial_services    | ChannelId_2 | -200   | 200   | 2018-11-16T08:16:36Z   | 2               | 0           |
| TransactionId_82586| BatchId_58031| AccountId_4841 | SubscriptionId_3829 | CustomerId_1001 | UGX           | 256         | ProviderId_4| ProductId_6| financial_services    | ChannelId_2 | -5000  | 5000  | 2018-11-16T08:16:40Z   | 2               | 0           |
| TransactionId_84144| BatchId_118083| AccountId_661 | SubscriptionId_4753 | CustomerId_1001 | UGX           | 256         | ProviderId_6| ProductId_10| airtime              | ChannelId_3 | 5200   | 5200  | 2018-11-16T08:20:39Z   | 2               | 0           |

---

## Preprocessing

### Label Encoding

Label encoding is applied to the `ProductCategory` column to convert the categorical values (such as "airtime" or "financial_services") into integer labels. This helps in cases where the order of categories matters or when you need a numerical representation of categorical data.

### One-Hot Encoding

One-hot encoding is applied to the `ProductCategory` and `ChannelId` columns. This method converts each category into a new binary column, indicating whether a specific category exists for each transaction.

For example:

| ProductCategory      | ChannelId   | ProductCategory_airtime | ProductCategory_financial_services | ChannelId_ChannelId_2 | ChannelId_ChannelId_3 |
|----------------------|-------------|-------------------------|------------------------------------|-----------------------|-----------------------|
| airtime              | ChannelId_3 | 1                       | 0                                  | 0                     | 1                     |
| financial_services   | ChannelId_2 | 0                       | 1                                  | 1                     | 0                     |

### Counting Value Occurrences

For each combination of `ProductCategory` and `ChannelId`, we calculate the count of occurrences, which helps us understand the distribution of categorical variables across the dataset.

---

## Usage

### Dependencies

Make sure you have the following dependencies installed:

- Python 3.x
- pandas
- numpy

Install required libraries via pip:

```bash
