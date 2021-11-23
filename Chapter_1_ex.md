# Chapter 1 Exercises

1. How would you define ML?
   1. The science (and art) of programming computers so they can learn from data.
2. List four types of problems where ML shines.
   1. Problems that have solutions with long lists of rules/specificities.
   2. Searching for insights about data.
   3. Environments that constantly get new data.
   4. When traditional approaches don't work (so no algo)
3. What's a labeled traing set?
   1. A training set used in supervised learning algorithms. These sets have what the associated "answer" is.
4. What are the two most common supervised tasks?
   1. Classification
   2. Targeting a numerical value
5. What are the four common unsupervised tasks?
   1. Association Rule Learning
   2. Visualization
   3. Clustering
   4. Dimensionality Reduction
6. What type of ML algorithm you use to traing a robot to walk in various unknown terrains?
   1. Reinforcement Learning, as we want it to function well in unfamiliar territory.
7. What type of algorithm would you use to segment customers into groups?
   1. Clustering algorithm if you don't know groups. Otherwise,a classification alogrithm would work.
8. Would you frame the problem of spam detection as a supervised or unsupervised learning problem?
   1. Supervised, but I think it probably uses some unsupervised techniques under the hood.
9. What's an online learning system?
   1.  An online learning system is a system that constantly trains itself on new data instances. It trains on the fly, so is good for continuous streams of data
10. What's out-of-core learning?
    1.  This is an online learning system that is used when all of the data can't fit on the device's memory - data is fed in sequentially, as a stream.
11. What type of machine learning algorithm uses a similarity measure to make predictions?
    1.  Instance based algorithms use similarity measures.
12. What is the difference between a model parameter and a hyperparameter?
    1.  A hyperparameter is a parameter of the learning model, not the algoritm. It affects its *learning rate* - i.e. how sensitive it is. The larger this is, the less sensitive your model is to new instances (flatter). It tells the algo how much regularization to apply.
    2.  A model parameter is a parameter that determines what the model predicts. The learning algorithm tries to find the optimal model parameters.
13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?
    1.  They try to optimize a utility/cost function. They look for the model parameters that generalize the best, then you feed the new instance's features into the model prediction function, using parameters found by the learning algorithm.
14. What are four main challenges of ML?
    1.  Bad Data
    2.  Nonrepresentative Training Data
    3.  Complex models/overfitting
    4.  Simple models
    5.  Uninformative Features
15. If your model performs well on training data, but does not generalize well, what is happening? What are three possible solutions?
    1.  You're overfitting your data. Some solutions:
        1.  simplifying the model via hyperparameter tuning or picking a simpler algo
        2.  getting more data
        3.  reducing data noise
16. What is a test set, and why would you want to use it?
    1.  A test set is a portion of your training data that you set aside to test the accuracy of the model that was trained on another set of the data. You use it to get an idea of how well your model is doing with a limited amount of data.
17. What is a validation set?
    1.  A validation set is a set that you leave aside to test which hyperparameter + model performs best on it. Think of it as another layer of testing.
18. What is the train-dev set, when do you need it, and how do you use it?
    1.  The train-dev set is another set of data that you use when there's a risk of mismatch between training data and data in validation and test sets. This is held out from the training set. You test the performance on this set after testing performance on the test set. If it performs consistently, then the issue is not overfitting, but a data mismatch - in this case, try to make the training data more representative.
19. What can go wrong if you tune hyperparameters on the test set?
    1.  You run the risk of overfitting to the test set. The generalization error measured will be too low as a result.