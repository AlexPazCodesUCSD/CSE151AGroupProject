# REPORT FOR AMAZON AGENT RATING PREDICTION MODEL

## INTRODUCTION
For our project we decided to go with being able to predict an amazon delivery drivers agent based off of several metrics. We chose to predict on this because a delivery drivers rating is one of the main differentiating factors between other delivery drivers, so we wanted to see just how much weight should be place on those ratings when considering the optimal delivery driver. Our project is cool because it really gives a good understanding of how the rating system actually works and what metrics are most important, like how agent age is actually the 2nd most correlated with agent rating. The broader impact of having a good predictive model of Agent Rating is companies and drivers will know what to prioritize for better customer reviews, and the factors in what may cause one driver to be more suited for a job over another.
## FIGURES

### Distribution of Delivery Time
![Distribution of Delivery Time](Distribution%20of%20Delivery%20Time.png)

## METHODS

### DATA EXPLORATION 
The code performs several data exploration steps: 
#### Distribution of Delivery Time
- A histogram is plotted to visualize the distribution of delivery times.
![Reference](#DISTRIBUTION_OF_DELIVERY_TIME)

### Delivery Time vs Agent Rating (Methods Section)
 ![Delivery Time vs Agent Rating](Delivery%20Time%20vs.%20Agent%20Rating.png)

### Impact of Weather on Delivery Time (Methods Section)
![Impact of Weather on Delivery Time](Impact%20of%20Weather%20on%20Delivery%20Time.png)

### Distribution of Agent Ratings (Methods Section)
![Distribution of Agent Ratings](Distribution%20of%20Agent%20Ratings.png)

### Correlation Heatmap (Methods Section)
![Correlation Heatmap](Correlation%20Heatmap.png)
 
### Boxplot of Delivery Time (Methods Section)
![Boxplot of Delivery Time](Boxplot%20of%20Delivery%20Time.png)

### Distance vs Delivery Time (Methods Section)
![Distance vs Delivery Time](Distance%20vs%20Delivery%20Time.png)

### Delivery Season vs Delivery Time (Methods Section)
![Delivery Season vs Delivery Time](Delivery%20Season%20vs%20Delivery%20Time.png)

### Pair Plots (Methods Section)
![Pair Plots](Pair%20Plots.png)

## RESULTS

### Actual vs Predicted Agent Ratings (Model 1 Linear Regression)
![Actual vs Predicted Agent Ratings](Actual%20vs%20Predicted%20Agent%20Ratings.png)

### Training vs Test Scores (Model 1 Linear Regression)
![Training vs Test Scores](Training%20vs%20Test%20Scores.png)

## DISCUSSION

## CONCLUSION
Due to our dataset having Agent Ratings that only went from 2.5 to 5.0 and the lionshare of that being from 4.0 and up it made it so our predictions would mostly be in that area, and mostly be correct after a simple amount of work was done. So it did not feel too satisfying to have a very accurate model. To combat that future models could get a more proportionate dataset, scale the Agent Ratings differently, or penalize getting the lower appearing Agent Ratings wrong moreso than it is already penalized. 

## STATEMENT OF COLLABORATION
