ok this is how the incremental learning framework works:

- ok we basically keep most of the class incremental learning framework the same (ensemble of binary classifiers for each activities) - use co-occurence detection to guide model updates when adding a new activity.

- the novel part is we are adding a sensor incremental learning component to the framework. the framework need to be able to handle 1) the addition of new sensors, and 2) missing sensors.

- these are the assumption we have for the framework:
    - start with n activities and m sensors.
    - participants can add activity at each timestep 1,2,3,...,t -> we have activity n+1, n+2, ..., n+t
    - at timestamp T, participants can add new sensors so activity n+T onwards would have full training data with m+1 sensors, but activity 1 to n+T-1 would have missing data for the new sensor. however, we want the model to still take in m+1 sensors as input and make predictions for all activities 1 to n+t.
    - this is the assumption on the participants end: for each activities, we will have a few samples with labeled data for the new activity.
    - however, after the addition of the new sensor, we will have a lot of unlabeled data with m+1 sensors.
    - we will use this data as samples to train a masked encoder to learn the physics of how m+1 sensors interact with each other. this will allow us to impute the missing sensor data for activities 1 to n+T-1, and also learn better representations for all activities with m+1 sensors.
    - we would keep training the masked encoder with the new unlabeled data as we collect more data with m+1 sensors, which will allow us to continuously improve the imputation and representation learning for all activities.
    - we will impute the previous labeled samples of activities 1 to n+T-1 with the masked encoder to create pseudo-labeled data with m+1 sensors, and use this data to train the binary classifiers for all activities 1 to n+t. this way we can leverage the new sensor data to improve the performance of the model on all activities, even those that were added before the new sensor was introduced.