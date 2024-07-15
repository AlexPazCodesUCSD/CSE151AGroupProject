# ABSTRACT
Our project aims to use the Amazon Delivery Dataset to predict the ratings of Amazon delivery agents using machine learning algorithms and techniques. The dataset encompasses over 43,632 delivery records across multiple cities, detailing various attributes such as order details, delivery agents' demographics, latitudinal and longitudinal coordinates of stores and delivery locations, order and delivery timestamps, and external factors like weather and traffic conditions. By analyzing these features, the project aims to identify patterns and correlations that influence delivery agents' performance ratings.
The objective is to build and evaluate predictive models that can accurately estimate an agent's rating based on the provided attributes. This analysis will not only enhance the understanding of key factors affecting delivery efficiency but also contribute to the improvement in logistical operations.

Dataset: https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset

# MILESTONE 2   
## Data Exploration

The dataset used in this project consists of various attributes related to Amazon deliveries. The exploration step involves evaluating the data, understanding the number of observations, data distributions, scales, missing data, and column descriptions.

### Data Summary

- **Number of Observations:** The dataset contains several thousand rows and multiple columns.
- **Column Descriptions:** 
  - Detailed descriptions of each column can be found in the notebook. 
  - Example columns include delivery time, agent ID, and customer rating.
- **Missing Data:** 
  - Missing data was identified and handled appropriately. 
  - Strategies included filling missing values with the mean, median, or mode of the column, or dropping rows/columns with excessive missing values.

### Data Distributions

- **Numerical Data:** 
  - Histograms and box plots were created for numerical columns to understand their distributions.
  - For example, delivery times and agent ratings were visualized to check for normality and outliers.
- **Categorical Data:** 
  - Bar charts were plotted for categorical columns to visualize the frequency of each category.
  - Examples include the frequency of different delivery locations or types of delivery vehicles.

## Data Visualization

Visualization is crucial for understanding the relationships between different attributes in the dataset.

### Scatter Plots

Scatter plots were used to visualize relationships between pairs of numerical attributes. This helped in identifying trends and correlations that are important for predictive modeling.

### Example Visualizations

Example visualizations were included to demonstrate the variety of data present. This included visualizing the number of deliveries per agent, average ratings, and more.

## Data Preprocessing

Data preprocessing is essential for preparing the data for modeling. The preprocessing steps for this project include:

1. **Handling Missing Data:** 
   - Missing values were filled using appropriate strategies such as the mean for numerical data and the mode for categorical data.
2. **Feature Scaling:** 
   - Standardization was applied to numerical features to ensure they have a mean of 0 and a standard deviation of 1.
3. **Encoding Categorical Data:** 
   - One-hot encoding was used for categorical variables to convert them into a numerical format suitable for machine learning models.
4. **Data Splitting:** 
   - The dataset was split into training and testing sets to evaluate model performance accurately.

## QUICK OVERVIEW
First. we clean up the dataset. We identify columns with missing values and count the number of missing values in each, print out the columns with missing values and the count of missing values in each, and cleaning the data by dropping rows with missing values. After cleaning up the dataset, we start to creating more features so that it is more easier for us to analyze the data and for people to visualize the pattern. We use store location, including store latitude and store longitude, and drop location, including drop latitude and drop longitude, to calculate the distance between store and drop-off location. We add this feature into our dataset. We also add a delivery season feature into our dataset, which splits delivery months into four seasons, which are spring, summer, fall, and winter. We have also changed the datetime formats for order date, order time, and pickup time so that the datetime structure is consistent in the dataset. The date format we use in the dataset is Year/Month/Date, and the time format we use in the dataset is Hour/Minute/Second.

Jupyter Notebook: https://colab.research.google.com/drive/1jelN7LeCg5STn4K33Uw23ikBc56peYlP#scrollTo=z_5LrR1wAFfD
