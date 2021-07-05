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
