# Chapter 1 - The Machine Learning Landscape

# What is Machine Learning?

Machine Learning is the science (and art) of programming computers so they can learn from data.

## What applications work well for Machine Learning?

* Problems where existing solutions need lots of fine tuning or long lists of rules - one ML algo can often simplify code *and* perform better.
* Complex problems where a traditional approach yields no good solution: the best ML techniques can maybe give you one
* Fluctuating Environments - ML techniques can adapt to new data.
* Getting insights about complex problems and large amounts of data.

# Types of ML systems

Several types!

### Supervised Learning
The training set you feed into the algorithm includes labels (the desired solutions).

A typical task here is *classification* - can assign occurrences into *classes*. Can also be used to predict a *target* numerical value, given a set of *features*.

### Unsupervised Learning

Training data is not labeled - system tries to learn without a teacher. Clustering, anomaly detection, visualization (algos that try to present data in a way that is understandable for humans), dimensionality reduction (algos to reduce complexity without losing info), and association rule learning (dig into a lot of data to find interesting relations between attributes) fit into this category.

### Semisupervised Learning

These algos are used in cases where you have partially labeled data. These are combinations of supervised and unsupervised algorithms. For example, *deep belief networks* (DBNs) are based on unsupervised components called *restricted Boltzmann machines* (RBMs) stacked on top of one another. These are trained sequentially in an unsupervised manner, and then the entire system is fine-tuned used supervised learning techniques.

### Reinforcement Learning 
THe learning system, known as the *agent* can observe its environment, select and perform actions, and earn rewards or *penalties* in return. It tries to learn the best strategy by itself. Think of AlphaGo. This case is interesting bc it wasn't learning when it played the champ - just relied on its training.

## Batch vs. Online learning

### Batch Learning
Does not learn from a stream of incoming data - mst be trained using all available data. This can take a lot of data + resources, so this is done offline. It just applies what it has learned, and runs without learning more.

Need to stop to train on new data, but this isn't too bad bc the ML process can be automated fairly easily. But there's still downtime - if you want something reactive, consider another approach. This approach can also be restrictive on cost + resources since you need to carry around a lot of data.

### Online learning
You train the system incrementally by feeding it data instances sequentially, either individually or in groups known as *mini-batches*. Learning steps are small and cheap, so you can learn on the fly.

Great for problems that receive a continuous flow of data and need to adapt quickly. Also good if you have limited resources - once an online algo has learned, can throw out the data.

Can also be used for *out-of-core* learning, or learning where the datasets can't fit in the machine's memory. Note that this is usually done not on the live system (so technically offline).

*Learning Rate* - how fast your system should adapt to changing data. If this is high, then your system will be sensitive to the most recent data

Downside to online systems is that if you feed it in bad data, its performance will gradually decline. 

## Instance vs. Model-Based Learning

One other way to categorize algorithms is by how they generalize. ML systems look a lot of data to make predictions. They need to *generalize* to, or make good predictions for, new examples they haven't seen before. Performance is great, but the goal is to perform well on new instances.

### Instance-based learning

Learning by heart and generalizing to new examples. For example, with a spam filter, you flag all emails identical to messages flagged by users. An alternative here is to find similar messages via a measure of similarity.

### Model-based learning

Build a model from the set of examples, and use that to make a prediction. For example, linear regression.

Before you use a model, you need to define your parameter values. In the case of a linear model, these are $\theta$ and $\theta_0$. The process of figuring out the values which will result in the best performance, you can define either a *utility function* that tells you how **good it is**. Or a *cost function* that tells you how **bad it is**. For linear regression, you typically use a cost function that measures the distance between linear model predictions and training examples.

Now, you feed your linear regression model your training data and it finds the best parameters - *training the model*.

At this point, he offers an example with a linear regression model and shows KNN as an instance model.

# Main Challenges of ML

Your main task is to select a learning algorithm and train it on data. The two things that can go wrong are bad data and bad algorithms.

## Bad Data

Some interesting papers out there show that several models perform similarly well on large datasets (for natural language problems). But not every dataset is large.

### Nonrepresentative Training Data

To generalize well, your training data must be representative of the new cases you want to generalize to. In the context of the happiness example, he omitted some countries in the original set. Adding those back in shows that there's not a perfect correlation between wealth and happiness.

It's hard to pick the right training set - if the sample is too small, you can have a lot of *sampling noise* (nonrepresentative data as a result of change). Large samples can be nonrepresentative if the sampling method is flawed (*sampling bias*). Some examples of sampling bias include the issues with phone polls - they only poll people that have phones and will pick them up.

### Poor Quality Data

Obv, having data riddled with errors, outliers, and noise will make it harder for the system to find underlying patterns. It's worth it to spend some time cleaning your data. This might include:

* Removing or fixing clear outliers
* How to deal with empty features in instances (e.g., some people not filling out age)

### Irrelevant Features

Garbage in, garbage out - need to figure out which features are best to train on. Picking these features is a process known as *feature engineering*. It involves a few steps:

* *Feature selection* - selecting the most useful features to train on among existing features
* *Feature extraction* - combining existing features to create a more useful one
* Creating new features by extracting more data

## Bad Algorithms

### Overfitting Training data

Self explanatory - performing well on training data but does not generalize well. This happens when the model is too complex relative to the amount and noisiness of the training data.

To deal with this, you can simplify the model by selecting fewer parameters, reducing the number of attributes, constraining the model, gathering more training data, or reducing noise in the training data.

*Constraining* the model to make it simpler and reducing the risk of overfitting is known as *regularization*. For example, in the linear model, we could fix $\theta_0 = 0$. Then, the model could only adjust $\theta$, making it simpler. We could also instead tell it to limit parameters to a certain range, making its degrees of freedom between one and two. 

The amount of regularization is determined by a parameter known as the *hyperparameter*. This is a parameter of a learning algorithm, not the model. So it's set prior to training and remains constant throughout training. If you make this constant large, then you'll get a flat model (slope close to 0) and 0 chance of overfitting; that said, you likely won't get a good solution. Tuning these is important!!

### Underfitting Training data

Opposite of overfitting! Some strategies for tackling underfitting:

* Select a more powerful model, with more parameters
* Feed better features to the learning algo.
* Reduce model constraints (lower regularization hyperparameter)

# Testing and Validating 

The only way to know how well a model generalizes is to try out new cases. Could do so by putting this into production, but not great for your end users. 

A better idea is to split your data into two sets - *training* and *test* sets. You train your model on training sets, and you test it on the test set. This error is known as *generalization error* (or *out-of-sample error*). Evaluating on test sets approximates this error. If your training error is low, but generalization error is high, **then your model is overfitting the data.**

## Hyperparameter Tuning and Model Selection

When you pick a hyperparameter, you don't want to pick the hyperparameter for *the set that you test on*! That overfits to your testing data.

A common solution here is *holdout validation*. With this, you leave out one of the training sets to evaluate several candidate models and select the best one. This held out set is called the *validation set*. You train multiple models on every set besides the validation set, and select the model that performs best on the validation set. Then, you train on your entire training set (including the validation set) to get your final model. Then, test this model on the test set to get generalization error.

Validation set size matters a lot here, so a good approach is *cross-validation*, where you use several small validation sets. You measure performance once perf validation set after training on the rest of the data. You take an average of performance to pick the best one. Downside here is that you need to train the model several times depending on the number of validation sets.

## Data Mismatch

You can get a lot of training data, but it may not be representative of the data that's used in production. 

In these cases, remember that it is most important that the validation set and test set must be as representative as the data that you expect to use in production, so they should almost completely be composed of representative data.

Still, it can be hard to know if the cause of poor performance is training overfitting or a data mismatch. You can attack this with another set, the *train-dev set*. After the model is trained on the training set, you evaluate it on *train-dev*.  If it performs well, then it's not overfitting. If it performs poorly on the validation set, it must be coming from a data mismatch.

# No Free Lunch Theorem

Models are simplified versions of observations. Simplifications are meant to discard superfluous observations not likely to generalize to new instances. Asa a result, you must make assumptions on what data to keep/discard. 

There's a famous paper that states if you make no assumptions about the data, there's no reason to prefer one model over another. This is known as the "No Free Lunch" theorem.  There is no model that is guaranteed to work best on every system. The only way to know is to evaluate all of them. 