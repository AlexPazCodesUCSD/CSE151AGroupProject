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

### <u>DATA EXPLORATION</u> 
The code performs several data exploration steps: 

#### Distribution of Delivery Time
- A histogram is plotted to visualize the distribution of delivery times. For more details, see [Figure 1](#fig1)
- ```plt.figure(figsize=(10, 6))
     sns.histplot(data['Delivery_Time'], kde=True)
     plt.title('Distribution of Delivery Time')
     plt.xlabel('Delivery Time (minutes)')
     plt.ylabel('Frequency')
     plt.show()


#### Relationship between Delivery Time and Agent Rating
- A scatter plot is used to explore the relationship between delivery time and agent ratings. For more details, see [Figure 2](#fig2)
- ```plt.figure(figsize=(10, 6))
     sns.scatterplot(x='Agent_Rating', y='Delivery_Time', data=data)
     plt.title('Delivery Time vs. Agent Rating')
     plt.xlabel('Agent Rating')
     plt.ylabel('Delivery Time (minutes)')
     plt.show()

#### Impact of Weather on Delivery Time
- A box plot shows how different weather conditions affect delivery times. For more details, see [Figure 3](#fig3)
- ```plt.figure(figsize=(10, 6))
     sns.boxplot(x='Weather', y='Delivery_Time', data=data)
     plt.title('Impact of Weather on Delivery Time')
     plt.xlabel('Weather')
     plt.ylabel('Delivery Time (minutes)')
     plt.show()

#### Distribution of Agent Ratings
- A count plot displays the distribution of agent ratings. For more details, see [Figure 4](#fig4)
- ```plt.figure(figsize=(10, 6))
     sns.countplot(x='Agent_Rating', data=data)
     plt.title('Distribution of Agent Ratings')
     plt.xlabel('Agent Rating')
     plt.ylabel('Count')
     plt.show()

#### Correlation Heatmap
- A heatmap is generated to visualize correlations between numerical features, including one-hot encoded features. For more details, see [Figure 5](#fig5)
- ```plt.figure(figsize=(12, 8))
     corr_matrix = numerical_data.corr()
     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
     plt.title('Correlation Heatmap (Including One-Hot Encoded Features)')
     plt.show()

#### Additional Visualizations
- Scatter plots, box plots, and pair plots are used to explore relationships between various features and delivery time. For more details, see [Figure 6](#fig6) , [Figure 7](#fig7) , [Figure 8](#fig8) , [Figure 9](#fig9)
- ```plt.figure(figsize=(10, 6))
     sns.scatterplot(x='Distance_Miles', y='Delivery_Time', data=data)
     plt.title('Distance vs Delivery Time')
     plt.xlabel('Distance (Miles)')
     plt.ylabel('Delivery Time (minutes)')
     plt.show()
     
     plt.figure(figsize=(10, 6))
     sns.boxplot(x='Delivery_Season', y='Delivery_Time', data=data)
     plt.title('Delivery Season vs Delivery Time')
     plt.xlabel('Delivery Season')
     plt.ylabel('Delivery Time (minutes)')
     plt.show()
     
     sns.pairplot(data[['Agent_Age', 'Agent_Rating', 'Distance_Miles', 'Delivery_Time']])
     plt.show()

### <u>**PREPROCESSING**</u>

#### Time-based Features
- Hour of the day is extracted from 'Order_Time' and 'Pickup_Time'.
- A feature is created for the time elapsed between order and pickup.
- ```data['Order_Hour'] = pd.to_datetime(data['Order_Time'], format='%H:%M:%S').dt.hour
     data['Pickup_Hour'] = pd.to_datetime(data['Pickup_Time'], format='%H:%M:%S').dt.hour
     
     data['Order_Pickup_Time_Diff'] = (pd.to_datetime(data['Pickup_Time'], format='%H:%M:%S') - 
                                        pd.to_datetime(data['Order_Time'], format='%H:%M:%S')).dt.total_seconds() / 60


#### Categorical Feature Encoding
- One-hot encoding is applied to 'Weather', 'Traffic', 'Vehicle', 'Area', and 'Category' columns.
- ```if all(col in data.columns for col in ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']):
         data = pd.get_dummies(data, columns=['Weather', 'Traffic', 'Vehicle', 'Area', 'Category'])
     else:
         print("Warning: One or more categorical columns not found. Skipping one-hot encoding.")

#### Outlier Detection and Handling
- A boxplot is used to visualize potential outliers in 'Delivery_Time'.
- Outliers are handled using the IQR method.
- ```plt.figure(figsize=(10, 6))
     sns.boxplot(data['Delivery_Time'])
     plt.title('Boxplot of Delivery Time')
     plt.show()

     Q1 = data['Delivery_Time'].quantile(0.25)
     Q3 = data['Delivery_Time'].quantile(0.75)
     IQR = Q3 - Q1
     lower_bound = Q1 - 1.5 * IQR
     upper_bound = Q3 + 1.5 * IQR
     # data = data[(data['Delivery_Time'] >= lower_bound) & (data['Delivery_Time'] <= upper_bound)]

#### Feature Scaling
- Numerical features ('Agent_Age', 'Distance_Miles', etc.) are standardized using StandardScaler.
- ```from sklearn.preprocessing import StandardScaler
     scaler = StandardScaler()
     numerical_features = ['Agent_Age', 'Distance_Miles', 'Order_Hour', 'Pickup_Hour', 'Order_Pickup_Time_Diff']


#### Feature Engineering
- 'Day_of_Week' and 'Is_Weekend' features are created from 'Order_Date'.
- ```data['Day_of_Week'] = data['Order_Date'].dt.dayofweek
     data['Is_Weekend'] = data['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)

### <u>**MODELS**</u>

#### Linear Regression
- A linear regression model is trained to predict 'Agent_Rating' using 'Agent_Age', 'Distance_Miles', and 'Delivery_Time' as features.
- The model is evaluated using mean squared error (MSE) and R-squared on both training and test sets.
- ```from sklearn.model_selection import train_test_split
     from sklearn.linear_model import LinearRegression
     from sklearn.metrics import mean_squared_error, r2_score
     
     X = data[['Agent_Age', 'Distance_Miles', 'Delivery_Time']]
     y = data['Agent_Rating']
     
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     
     model = LinearRegression()
     model.fit(X_train, y_train)
     
     y_pred_training = model.predict(X_train)
     y_pred_test = model.predict(X_test)
     
     training_mse = mean_squared_error(y_train, y_pred_training)
     testing_mse = mean_squared_error(y_test, y_pred_test)
     r2_training = r2_score(y_train, y_pred_training)
     r2_testing = r2_score(y_test, y_pred_test)
     
     print(f"Training Mean Squared Error: {training_mse}")
     print(f"Training Mean Squared Error: {testing_mse}")
     print(f"R-squared Training: {r2_training}")
     print(f"R-squared Testing: {r2_testing}")
     print(f"Coeffecients: {model.coef_}")

#### Random Forest Regressor
- A Random Forest Regressor is trained with 100 estimators and a random state of 42.
- It uses a wider range of features including delivery time, distance, agent age, time differences, weather conditions, vehicle types, area types, and day of the week.
- The model's performance is evaluated using Mean Absolute Error (MAE) and Mean Squared Error (MSE) on the test data.
- ```from sklearn.ensemble import RandomForestRegressor
     from sklearn.metrics import mean_squared_error, mean_absolute_error
     
     features = ['Delivery_Time', 'Distance_Miles', 'Agent_Age', 'Order_Pickup_Time_Diff',
                 'Order_Hour', 'Pickup_Hour', 'Weather_Cloudy', 'Weather_Fog',
                 'Vehicle_motorcycle ', 'Vehicle_scooter ', 'Vehicle_van', 'Area_Metropolitian ',
                 'Area_Other', 'Area_Semi-Urban ', 'Area_Urban ', 'Day_of_Week', 'Is_Weekend']
     
     target = 'Agent_Rating'
     
     X = data[features]
     y  = data[target]
     
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     
     rf = RandomForestRegressor(n_estimators=100, random_state=42)
     rf.fit(X_train, y_train)
     
     y_pred = rf.predict(X_test)
     mae = mean_absolute_error(y_test, y_pred)
     mse = mean_squared_error(y_test, y_pred)
     print(f'Mean Absolute Error on Test Data: {mae}')
     print(f'Mean Squared Error on Test Data: {mse}')

#### Gradient Boosting Regressor
- A Gradient Boosting Regressor is used, and a GridSearchCV is employed to find the best hyperparameters from a defined parameter grid.
- The grid search considers different values for the number of estimators, learning rate, maximum depth, and subsample.
- The model with the best parameters (determined using 5-fold cross-validation and 'neg_mean_absolute_error' scoring) is then trained and evaluated using MAE and MSE on the test data.
- ```from sklearn.ensemble import GradientBoostingRegressor
     from sklearn.model_selection import GridSearchCV
     
     param_grid = {
         'n_estimators': [100, 200, 300],
         'learning_rate': [0.01, 0.1, 0.05],
         'max_depth': [3, 4, 5],
         'subsample': [0.8, 0.9, 1.0]
     }
     
     gbm = GradientBoostingRegressor(random_state=42)
     
     grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
     grid_search.fit(X_train, y_train)
     
     best_params = grid_search.best_params_
     print(f'Best parameters: {best_params}')
     
     best_gbm = GradientBoostingRegressor(**best_params, random_state=42)
     best_gbm.fit(X_train, y_train)
     
     y_pred = best_gbm.predict(X_test)
     mae = mean_absolute_error(y_test, y_pred)
     mse = mean_squared_error(y_test, y_pred)



## RESULTS

### Actual vs Predicted Agent Ratings (Model 1 Linear Regression)
![Actual vs Predicted Agent Ratings](Actual%20vs%20Predicted%20Agent%20Ratings.png)

### Training vs Test Scores (Model 1 Linear Regression)
![Training vs Test Scores](Training%20vs%20Test%20Scores.png)

## DISCUSSION

## CONCLUSION
Due to our dataset having Agent Ratings that only went from 2.5 to 5.0 and the lionshare of that being from 4.0 and up it made it so our predictions would mostly be in that area, and mostly be correct after a simple amount of work was done. So it did not feel too satisfying to have a very accurate model. To combat that future models could get a more proportionate dataset, scale the Agent Ratings differently, or penalize getting the lower appearing Agent Ratings wrong moreso than it is already penalized. 

## STATEMENT OF COLLABORATION
