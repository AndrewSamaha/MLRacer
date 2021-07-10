# MLRacer
This project is inspired by the work of Assistant Professor [Madhur Behl](https://engineering.virginia.edu/faculty/madhur-behl) in the Department of Computer Science at the University of Virginia and uses a modified version of [HueRacer](https://github.com/KilledByAPixel/HueJumper2k) by [Frank Force](https://github.com/KilledByAPixel).

![Original Video of Modified HueRacer](https://github.com/AndrewSamaha/MLRacer/blob/main/img/original.gif?raw=true)
![Deltas Video of Modified HueRacer](https://github.com/AndrewSamaha/MLRacer/blob/main/img/deltas.gif?raw=true)


# Why Autonomous Racing?
Training machines to be agile and performant at their physical limits can improve safety in public roads through both autonomous and driver-assisted systems that can avoid accidents. 

# Perception
Autonomous driving systems need to solve three problems:
1. Perception
1. Planning
1. Control

Work by Behl has shown that simple (alghorithmic as opposed to deep-learning) approaches to control can produce AI drivers as good as the best human drivers once perception has been solved. Therefore, this project focuses on perception. That is given a driver's point-of-view, can a model be trained to identify the vehicles position, speed, and trajectory on the road? 

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

The good news is that nothing unusual is happening here -- the ranges are exactly what I expected. And, that's also the bad news. Why does the network perform more poorly when the values are normalized?

![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha2.position.png?raw=true)

## Effect of Normalization

Normalized Inputs (50 epochs):

![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.normalized.50epochs.png?raw=true)

Unnormalized Inputs (50 epochs):

![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.unnormalized.50epochs.png?raw=true)

Unnormalized Inputs (1000 epochs):

![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.unnormalized.1000epochs.png?raw=true)

## SGD - no worky
## RMSProb

Unnormalized Inputs (100 epochs):

![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.rmsprop.unnormalized.100epochs.png?raw=true)

Unnormalized Inputs (1000 epochs):

![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.rmsprop.unnormalized.1000epochs.png?raw=true)

## Adadelta
It's good to know what doesn't work, right?
1000 epochs...
![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.adadelta.unnormalized.1000epochs.png?raw=true)


## Adagrad
The interesting thing here is that each epoch produced a very consistent decrease in error. So this has some potential to do better given more epochs. I also like the shape of the distribution in the test set.
1000 epochs...
![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.adagrad.unnormalized.1000epochs.png?raw=true)

## Adamax
Adamax bottomed out pretty quickly. Might benefit from a lower learning rate.
1000 epochs...
![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.adamax.unnormalized.1000epochs.png?raw=true)

## Nadam
Nadam bottomed out pretty quickly. Might benefit from a lower learning rate.
1000 epochs...
![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.nadam.unnormalized.1000epochs.png?raw=true)

## Experiments with mean squared logarithmic error

![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.rmsprop.msle.unnormalized.1000epochs.png?raw=true)

## Experiments with mean absolute error

![Comparison of Training and Test](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.rmsprop.mae.unnormalized.1000epochs.png?raw=true)

# 6-Layer Model
    model = Sequential()
    model.add(Flatten(input_shape=(2500,1)))
    model.add(Dense(625, activation="relu"))
    model.add(Dropout(.1))
    model.add(Dense(160, activation="relu"))
    model.add(Dropout(.1))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.MeanSquaredError()])
    model.build()

## 10% dropout
100 epochs-
![6 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.6layer.10pctdrop.rmsprop.mae.unnormalized.100epochs.png?raw=true)

## 20% dropout
100 epochs-
![6 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.6layer.20pctdrop.rmsprop.mae.unnormalized.100epochs.png?raw=true)

## 30% dropout
100 epochs-
![6 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.6layer.30pctdrop.rmsprop.mae.unnormalized.100epochs.png?raw=true)

## 30% dropout, .0005 learning rate
100 epochs-
![6 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.6layer.30pctdrop.rmsprop0005.mae.unnormalized.100epochs.png?raw=true)

## 3rd 30% dropout layer, .00005 learning rate
100 epochs-
![6 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.6layer.3-30pctdrop.rmsprop00005.mae.unnormalized.100epochs.png?raw=true)

# 5-layer Model, 30% dropout, mse
    model = Sequential()
    model.add(Flatten(input_shape=(2500,1)))
    model.add(Dense(160, activation="relu"))
    model.add(Dropout(.3))
    model.add(Dense(40, activation="relu"))
    model.add(Dropout(.3))
    model.add(Dense(10, activation="relu")) # lets try an intermediary layer of 625?
    model.add(Dense(1))
    #adam vs. sgd; adam works better
    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[tf.keras.metrics.MeanSquaredError()])
    model.build()

    train_rsquared, test_rsquared, modelPosition  = buildModelSingleTask(X_train, y_train_position.reshape(-1,1), X_test, y_test_position.reshape(-1,1), optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.00005), loss='mean_squared_error', inputScaler=1, epochs=100, filename="alpha.5layer.2-30pctdrop.rmsprop00005.mse.unnormalized.100epochs.png")

# 5-layer Model, 40% dropout, mse
![5 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.5layer.2-30pctdrop.rmsprop00005.mse.unnormalized.100epochs.png?raw=true)

    model = Sequential()
    model.add(Flatten(input_shape=(2500,1)))
    model.add(Dropout(.4))
    model.add(Dense(160, activation="relu"))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(10, activation="relu")) # lets try an intermediary layer of 625?
    model.add(Dense(1))
    #adam vs. sgd; adam works better
    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[tf.keras.metrics.MeanSquaredError()])

    train_rsquared, test_rsquared, modelPosition  = buildModelSingleTask(X_train, y_train_position.reshape(-1,1), X_test, y_test_position.reshape(-1,1), optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.00005), loss='mean_squared_error', inputScaler=1, epochs=100, filename="alpha.5layer.1-40pctdrop.rmsprop00005.mse.unnormalized.100epochs.png")


![5 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.5layer.1-40pctdrop.rmsprop00005.mse.unnormalized.100epochs.png?raw=true)

# 5-layer Model, 50% dropout, mse
![5 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.5layer.1-50pctdrop.rmsprop00005.mse.unnormalized.100epochs.png?raw=true)

# 5-layer Model, 60% dropout, mse
![5 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.5layer.1-50pctdrop.rmsprop00005.mse.unnormalized.100epochs.png?raw=true)

# 4-layer Super Dense, 20% dropout, mse
    model = Sequential()
    model.add(Flatten(input_shape=(2500,1)))
    model.add(Dropout(.2))
    model.add(Dense(2500, activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(2500, activation="relu"))
    model.add(Dense(1))
    #adam vs. sgd; adam works better
    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[tf.keras.metrics.MeanSquaredError()])
    model.build()

    train_rsquared, test_rsquared, modelPosition  = buildModelSingleTask(X_train, y_train_position.reshape(-1,1), X_test, y_test_position.reshape(-1,1), optimizer= tf.keras.optimizers.RMSprop(), loss='mean_squared_error', inputScaler=1, epochs=100, filename="alpha.4layer.2-20pctdrop.rmsprop.mse.unnormalized.100epochs.png")

100 epochs...
![4 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.4layer.2-20pctdrop.rmsprop.mse.unnormalized.100epochs.png?raw=true)

Finally, some promising results on the test dataset. What happens if we feed the model model data?

# 4-layer Super Dense, 2-20% dropout layers, mse

100 epochs...

![4 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.4layer.2-20pctdrop.rmsprop.mse.unnormalized.24krows.100epochs.png?raw=true)

I was doing all this with a growing worry in the back of my mind that hadn't quite formed into words. A few hours after getting this result, the words finally came: the difference between one image and the next was not simply a function of the vehicle's speed, but also of 1/rate at which I was capturing frames. So, I needed to rewrite my process function to encode time deltas between images and then pass that into the network as input. So, my nice round number for input array of 2500 would be 2501. While this shouldn't matter for position (what the above model predicted), it would certainly become a problem later when trying to predict velocity.

# 4-layer Dense, 2500+1 columns
100 epochs...

![4 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.4layer.2-20pctdrop.rmsprop.mse.unnormalized.24krows.tdelta.100epochs.png?raw=true)

Here's what the performance looks like when plotted again actual position data using a holdout dataset of 5k rows:
![4 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.4layer.2-20pctdrop.rmsprop.mse.unnormalized.24krows.tdelta.100epochs.holdout.png?raw=true)

# 5-layer dense, 2500+1 columns, 40k rows

    model = Sequential()
    model.add(Flatten(input_shape=(2501,1)))
    model.add(Dropout(.2))
    model.add(Dense(2501, activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(2501, activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(2501, activation="relu"))
    model.add(Dense(1))
    #adam vs. sgd; adam works better
    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[tf.keras.metrics.MeanSquaredError()])
    model.build()
    model.summary()


    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 2501)              0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 2501)              0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 2501)              6257502   
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 2501)              0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 2501)              6257502   
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 2501)              0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 2501)              6257502   
    _________________________________________________________________
    dense_7 (Dense)              (None, 1)                 2502      
    =================================================================
    Total params: 18,775,008
    Trainable params: 18,775,008
    Non-trainable params: 0


Doing an experiment with a deeper network after listening to TWIML AI Podcast #378 with Joseph Gonzalez, Assistant Professor in the EECS Department of UC Berkeley
![5 Layer Model](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.5layer.3-20pctdrop.rmsprop.mse.unnormalized.40krows.100epochs.png?raw=true)

![5 Layer Model Across Time](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.5layer.3-20pctdrop.rmsprop.mse.unnormalized.40krows.tdelta.100epochs.holdout.png?raw=true)

![5 Layer Model Residuals](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.5layer.3-20pctdrop.rmsprop.mse.unnormalized.40krows.100epochs.residuals.png?raw=true)

![5 Layer Model Residuals Hist](https://github.com/AndrewSamaha/MLRacer/blob/main/img/alpha.5layer.3-20pctdrop.rmsprop.mse.unnormalized.40krows.100epochs.residuals.hist.png?raw=true)
