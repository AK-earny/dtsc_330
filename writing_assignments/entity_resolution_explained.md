# What is entity resolution?:
Entity resolution refers to the task of trying to idenitify the same entity from different datasets.

How has the work we've done in class related to solving entity resolution?:
Week 1: We talked about LLMs and how they work. We were also introduced to features and labels, integral parts of the entity resoluion progress as they bascially determine whether data relates or not.

Week 2: We fiddled with early ideas of entity resolution with the articles and grantees datasets as we organized the data into individual people. We talked about things like data mergeing and further fleshed out the ideas of week 1. We asked questions like: "How can we fix inconsistencies in the data set in order to have the datasets similarly organized?" If the data isn't clean than the model isn't gonna learn the things we want it to.

Week 3: We again built off the ideas of the last few classes, going into depth about the pipeline of machine learning and learning about classifiers which are essential to entity resolution. For homework we created classifiers for fitbit data, which helped us understand how models learn patterns from data to make predictions, mapping features to labels.

Week 4: We learned about the different machine learning models like XGBoost, decision trees, and random forest. We then used these models (mostly XGBoost) to improve our classifiers. 

Week 5: This is when we properly started getting into the problem of entity resolution. Kind of a review day of everything we just filled in the gaps relating to classifiers and features and labels. 

Week 6: We went over how to generate simulated training data when we lack proper data.

Week 7: With SQL we learned how to build and combine large datasets quickly.

While the last few weeks haven't been as "entity resolution" heavy, weeks 1-5 help us get a rough understaning of the process:
Data cleaning -> Features -> Labels -> Model training -> Prediction  
As I was looking through our past work, I noticed that the entity resolution pipeline closely follows the structure of the machine learning pipeline, thats not to say they are the same however. Entity resolution is a specific problem thats solved THROUGH machine learning, not machine learning itself.
