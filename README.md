# Fraud-Detection

Given fraud event detection data, we made a Gradient Boosting model which focused minimizing the number of missed fraud events, and used recall and F1 as our metrics.

After feature engineering and EDA (Note that EDA is missing due to the confidentiality of this data), we identified relevant features such as number of tickets sold, whether the event has a Facebook page, and event description body length were indicative of fraud. Notably, fraud was often associated with particularly low numbers (i.e. 0 means no Facebook page), while non-fraud events could have low or high values. Therefore, we determined tree based models would be most effective at making value distinctions with the greatest information gain. 

As fraud only constituted 9% of all events, our baseline predicted no fraud. Our logistic model had a F1 score of 72%, and Recall of 61%. Our Boosted model had a F1 score of 84% and Recall of 93%. We therefore confirmed that the Boosted model is the best for identifying fraud.
