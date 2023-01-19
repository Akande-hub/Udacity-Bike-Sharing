# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Akande Oluwatosin Adetoye

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
TODO: Some of the prediction contains negative prediction and Kaggle requires the predicted values to be non-negative, then we consider setting all negative predictions to 0 before submitting them to Kaggle in order for the submission to be accepted.
### What was the top ranked model that performed?
TODO:  According to the summary table of the fitted models, the best-performing model was "WeightedEnsemble_L3" with a score value of -52.923500 (root mean squared error). This model took about 500 seconds to train, the longest of all the models. It also had the lowest pred_time_val_marginal and a maximum stack level of 3. 

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
TODO:The exploratory data analysis of the Bike Sharing Demand project revealed that the train data contains 12 columns and 10,886 rows, while the test data contains 9 columns and 6,443 rows. All of the values in the columns are integer or floating point, with the exception of the "datetime" column, which is of datetime type. There are no missing or NaN values in either of the datasets. The histogram of the train data showed the distribution of each feature in the train data, and the correlation plot showed the relationships between the different features. The timeseries plot of the bike sharing demand showed a seasonal pattern in the train data, with demand for bike sharing increasing in a sinusoidal pattern over the years. We found that the "casual" column had a high overall correlation with several other columns, and that "casual" and "registered" were highly correlated and missing from the test data. We therefore decided to drop these columns, as strong correlations between features often add little or no new information to the model. We also found that the "datetime" column had a very high cardinality, but instead of dropping the feature, we decided to extract four additional features (year, month, day, and hour) from it in order to preserve the information it contained.

### How much better did your model preform after adding additional features and why do you think that is?
TODO: After adding additional features extracted from the "datetime" feature (such as day, month, year, and hour) and converting the "seasons" and "weather" features to categories, the model performance score increased from "-52.923500" to "-35.707278" and the Kaggle submission score improved significantly from around "1.8" to "0.47" (a 73% improvement). These changes helped the model to learn the data more effectively by providing more useful information and by allowing the model to better understand the specific characteristics of the "seasons" and "weather" features. As a result, the model's performance improved significantly.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
TODO: After testing different combinations of hyperparameters, we found that the "WeightedEnsemble_L3" model consistently performed the best. However, we also observed that hyperparameter tuning did not always lead to further improvements in the model's performance. In some cases, adjusting the hyperparameters actually resulted in worse performance. For instance, setting the "presets" parameter to "optimize_for_deployment" decreased the performance of the model, while increasing the "limit_time" parameter sometimes had mixed results..  

### If you were given more time with this dataset, where do you think you would spend more time?
TODO: If I had more time, I would have taken a deeper dive into the exploratory data analysis in order to identify additional features that could be added to the model. This would help the model to learn the characteristics of the data more effectively. Additionally, I would have spent more time on hyperparameter tuning to find the optimal configuration for the model. This would help to achieve even better performance by fine-tuning the model to the specific characteristics of the data.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|hpo|eval_metric: r2|limt_time:900,num_bag_sets=15|default val|0.44740|
|initial|default val|default val|default val|1.80422|
|add_features|default val|default val|default val|0.44416|
|hpo|default val|limt_time:900|default val|0.44331|

Available on the jupyter nootbook.
### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

Available on the jupyter nootbook.
### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)
Available on the jupyter nootbook.
## Summary
TODO:  In this case study, I developed an AutoGluon model to predict bike demand based on the given bike sharing data. I began by exploring and analyzing the features of the data, using visualization and statistical analysis tools such as describe and hist in pandas. Then, I built an initial model using the provided parameters and without any further feature engineering. The best-performing model was identified as "WeightedEnsemble_L3", and I used this model to make predictions and submit them to Kaggle to establish an initial benchmark score.

I the next phase I conduct an exploratory data analysis (EDA) in order to identify patterns and trends in the data. I used visualizations and other analysis techniques to gain insights into the data and inform decision-making. Based on the findings, I decided to transform some of the numeric features into categorical features and to create new features that captured information about the time for each datetime record. After implementing these changes, I trained the model again and found that the best-performing modele remain "WeightedEnsemble_L3" and the Kaggle score on the test data significantly improved by 73%.

After preprocessing the features, I decided to tune the hyperparameters for the model algorithm in order to improve its performance. I believed that we could achieve better scores by adjusting the hyperparameter settings and increasing the duration of model training.

Finally, I compared the Kaggle scores of all the trained models and plotted these scores against the different hyperparameter settings in order to analyze the relative impact on performance. This allowed me to identify the most effective hyperparameter configurations.


