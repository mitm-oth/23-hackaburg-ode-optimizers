# Hackaburg 2023 - ODEoptimizers

Data prediction of timeseries driving data with a ordinary differential equation (ODE)

<https://devpost.com/software/odeoptimizers>

## data insights

![](assets/2023-05-26-23-46-46.png)

Following the data to the Nordkapp

![](assets/2023-05-26-12-35-30.png)
<!-- ![](assets/2023-05-26-01-20-04.png) -->

## physical model

![](assets/2023-05-27-15-02-38.png)

## steps

* splitting data in time consistent chunks
* smoothing data  
  ![](assets/pngs/7a_raw_sensor.png)  
  ![](assets/pngs/7b_sensors.png)
* prediction  
  ![](assets/pngs/7c_prediction.png)
  

## results

![](assets/pngs/7d_sensor_vs_prediction.png)

Overfitting? No: We created a physical model that tries to align with the real world. It is not influenced by the training data.

=> Validation with other data

![](assets/pngs/58d_sensor_vs_prediction.png)

## further steps

* Use more data to find the coefficients
* Feed model with real time finetuned coefficients from a ML-model
* Car specific coefficients using a ML-model
* Test under less extreme driving conditions

<!-- ## tasks

* [X] Getting data insights
* [X] ~~Copying data into influx db~~Car specific coefficients
using a ML-model
* [X] Splitting data in test and training data
* [X] Choosing a model => ODE
* [X] Try AWS SageMaker Canvas, DataWrangler, Studio AutoML -->
