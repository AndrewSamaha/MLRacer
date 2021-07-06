# MLRacer
This project is inspired by the work of Assistant Professor [Madhur Behl](https://engineering.virginia.edu/faculty/madhur-behl) in the Department of Computer Science at the University of Virginia and uses a modified version of [HueRacer](https://github.com/KilledByAPixel/HueJumper2k) by [Frank Force](https://github.com/KilledByAPixel).

![Video of Modified HueRacer](https://github.com/AndrewSamaha/MLRacer/raw/main/original.gif)


# Why Autonomous Racing?
Training machines to be agile and performant at their physical limits can improve safety in public roads through both autonomous and driver-assisted systems that can avoid accidents. 

# Perception
Autonomous driving systems need to solve three problems:
1. Perception
1. Planning
1. Control

This project focuses on the first problem, perception. That is given a driver's point-of-view, can a model be trained to identify the vehicles position, speed, and trajectory on the road?

# Iteration
## Baseline
The goal was to start with a very simple NN. I chose a 3 layer NN (2500, 25-relu, and 3) and a mean-squared-error loss function because I was wanted to minimize the error between the actual and predicted values of 3 target tasks (velocity, position, and rotation). I evaluated the results across a range of 10 to 225 epochs and got both and flat r-squared values on the test dataset:

![Baseline alpha.v2](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.v2.png?raw=true)

## Single-Task
The previous model was multi-task: outputting three neurons, one for each of the following componets of trajectory: location, velocity, and rotation. Maybe multi-task models are too hard, or maybe it was actually good at one or two of the tasks and terrible at another. Hence, I looked at them separately:
train_rsquared, test_rsquared = buildModelSingleTask(X_train, y_train_position.reshape(-1,1), X_test, y_test_position.reshape(-1,1), inputScaler=1)
Training R-Squared: 0.9917317586090525
Test R-Squared: 0.4966193845385154

train_rsquared, test_rsquared = buildModelSingleTask(X_train, y_train_velocity.reshape(-1,1), X_test, y_test_velocity.reshape(-1,1), inputScaler=1)
Training R-Squared: 0.9657727443058403
Test R-Squared: 0.3707565479826129

train_rsquared, test_rsquared = buildModelSingleTask(X_train, y_train_rotation.reshape(-1,1), X_test, y_test_rotation.reshape(-1,1), inputScaler=1)
Training R-Squared: -0.015015659592109376
Test R-Squared: -0.0037359842824022937

## EDA- One step forward, two steps back
Goal: Understand the distributions of the target classes (position, velocity, & rotation) and see what the above baseline models are predicting.

![EDA of X values](https://github.com/AndrewSamaha/MLRacer/blob/main/img/x_eda.png?raw=true)
![EDA of y values](https://github.com/AndrewSamaha/MLRacer/blob/main/img/y_eda.png?raw=true)