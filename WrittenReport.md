# REPORT FOR AMAZON AGENT RATING PREDICTION MODEL

## INTRODUCTION
For our project we decided to go with being able to predict an amazon delivery drivers agent based off of several metrics. We chose to predict on this because a delivery drivers rating is one of the main differentiating factors between other delivery drivers, so we wanted to see just how much weight should be place on those ratings when considering the optimal delivery driver. Our project is cool because it really gives a good understanding of how the rating system actually works and what metrics are most important, like how agent age is actually the 2nd most correlated with agent rating. The broader impact of having a good predictive model of Agent Rating is companies and drivers will know what to prioritize for better customer reviews, and the factors in what may cause one driver to be more suited for a job over another.
## FIGURES
### Distribution of Delivery Time <a name="fig1"></a>
![Distribution of Delivery Time](Distribution%20of%20Delivery%20Time.png)

### Delivery Time vs Agent Rating <a name="fig2"></a>
![Delivery Time vs Agent Rating](Delivery%20Time%20vs.%20Agent%20Rating.png)


### Impact of Weather on Delivery Time <a name="fig3"></a>
![Impact of Weather on Delivery Time](Impact%20of%20Weather%20on%20Delivery%20Time.png)


### Distribution of Agent Ratings <a name="fig4"></a>
![Distribution of Agent Ratings](Distribution%20of%20Agent%20Ratings.png)


### Correlation Heatmap <a name="fig5"></a>
![Correlation Heatmap](Correlation%20Heatmap.png)

 
### Boxplot of Delivery Time <a name="fig6"></a>
![Boxplot of Delivery Time](Boxplot%20of%20Delivery%20Time.png)

### Distance vs Delivery Time <a name="fig7"></a>
![Distance vs Delivery Time](Distance%20vs%20Delivery%20Time.png)

### Delivery Season vs Delivery Time <a name="fig8"></a>
![Delivery Season vs Delivery Time](Delivery%20Season%20vs%20Delivery%20Time.png)

### Pair Plots <a name="fig9"></a>
![Pair Plots](Pair%20Plots.png)

## METHODS

### DATA EXPLORATION 
The code performs several data exploration steps: 

#### Distribution of Delivery Time
- A histogram is plotted to visualize the distribution of delivery times. For more details, see [Figure 1](#fig1)

#### Relationship between Delivery Time and Agent Rating
- A scatter plot is used to explore the relationship between delivery time and agent ratings. For more details, see [Figure 2](#fig2)

#### Impact of Weather on Delivery Time
- A box plot shows how different weather conditions affect delivery times. For more details, see [Figure 3](#fig3)

#### Distribution of Agent Ratings
- A count plot displays the distribution of agent ratings. For more details, see [Figure 4](#fig4)

#### Correlation Heatmap
- A heatmap is generated to visualize correlations between numerical features, including one-hot encoded features. For more details, see [Figure 5](#fig5)

#### Additional Visualizations
- Scatter plots, box plots, and pair plots are used to explore relationships between various features and delivery time. For more details, see [Figure 6](#fig6) , [Figure 7](#fig7) , [Figure 8](#fig8) , [Figure 9](#fig9)

### PREPROCESSING

#### Time-based Features
- Hour of the day is extracted from 'Order_Time' and 'Pickup_Time'.
- A feature is created for the time elapsed between order and pickup.

#### Categorical Feature Encoding
- One-hot encoding is applied to 'Weather', 'Traffic', 'Vehicle', 'Area', and 'Category' columns.

#### Outlier Detection and Handling
- A boxplot is used to visualize potential outliers in 'Delivery_Time'.
- Outliers are handled using the IQR method.

#### Feature Scaling
- Numerical features ('Agent_Age', 'Distance_Miles', etc.) are standardized using StandardScaler.

#### Feature Engineering
- 'Day_of_Week' and 'Is_Weekend' features are created from 'Order_Date'.

### MODELS

#### Linear Regression
- A linear regression model is trained to predict 'Agent_Rating' using 'Agent_Age', 'Distance_Miles', and 'Delivery_Time' as features.
- The model is evaluated using mean squared error (MSE) and R-squared on both training and test sets.

#### Random Forest Regressor
- A Random Forest Regressor is trained with 100 estimators and a random state of 42.
- It uses a wider range of features including delivery time, distance, agent age, time differences, weather conditions, vehicle types, area types, and day of the week.
- The model's performance is evaluated using Mean Absolute Error (MAE) and Mean Squared Error (MSE) on the test data.

#### Gradient Boosting Regressor
- A Gradient Boosting Regressor is used, and a GridSearchCV is employed to find the best hyperparameters from a defined parameter grid.
- The grid search considers different values for the number of estimators, learning rate, maximum depth, and subsample.
- The model with the best parameters (determined using 5-fold cross-validation and 'neg_mean_absolute_error' scoring) is then trained and evaluated using MAE and MSE on the test data.


## RESULTS

### Actual vs Predicted Agent Ratings (Model 1 Linear Regression)
![Actual vs Predicted Agent Ratings](Actual%20vs%20Predicted%20Agent%20Ratings.png)

### Training vs Test Scores (Model 1 Linear Regression)
![Training vs Test Scores](Training%20vs%20Test%20Scores.png)

## DISCUSSION

## CONCLUSION
Due to our dataset having Agent Ratings that only went from 2.5 to 5.0 and the lionshare of that being from 4.0 and up it made it so our predictions would mostly be in that area, and mostly be correct after a simple amount of work was done. So it did not feel too satisfying to have a very accurate model. To combat that future models could get a more proportionate dataset, scale the Agent Ratings differently, or penalize getting the lower appearing Agent Ratings wrong moreso than it is already penalized. 

## STATEMENT OF COLLABORATION
