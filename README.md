# Machine-Learning-Approaches-for-Static-Modeling-of-SOFC-Using-GP-and-ANN

## Project Description

This project aims to solve the static modeling problem addressed in the [Energy 2009 paper](https://www.sciencedirect.com/science/article/abs/pii/S0360544209000449). The primary goal is to use Genetic Programming (GP), Machine Learning, and Artificial Neural Networks (ANN) to create a model for Solid Oxide Fuel Cells (SOFC) that is more efficient and accurate than the model presented in the 2009 paper. We use the same SOFC model and create training and test data as in that paper but with the freedom to use any version of GP and any type of ANN (not necessarily RBF). The aim is to achieve statistically significantly better results compared to those reported in the original paper.

## Algorithm Parameters

### SOFC Parameters

| Parameter         | Value              |
|-------------------|--------------------|
| Number of Stacks  | 1                  |
| Temperature Values| 100                |
| E0                | 0.8                |
| N0                | 0.01               |
| u                 | 0.8                |
| Kr                | 0.993e-3 # mol/(s A) |
| Kr_cell           | Kr / N0            |
| KH2               | 0.843 # $mol/(s atm)$ |
| KH2O              | 0.281 # $mol/(s atm)$ |
| KO2               | 2.52 # $mol/(s atm)$  |
| r                 | 2.52 # $mol/(s atm)$  |
| rcell             | r / N0             |
| rHO               | 1.145              |
| i0_den            | 20 # $mA/cm2$        |
| ilimit_den        | 900 # $mA/cm2$       |
| A                 | 1000 # $cm2$         |
| n                 | 2                  |
| F                 | 96485              |
| alpha             | 0.5                |
| R                 | 0.0821 # $atm/(mol K)$ |
| N                 | 20                 |
| PH2               | 1.265              |
| PO2               | 2.527              |
| PH2O              | 0.467              |

### GP Model Parameters

| Parameter               | Value  |
|-------------------------|--------|
| Number of Independent Runs | 50   |
| Number of Generations   | 50     |
| Population Size         | 100    |
| Tournament Size         | 3      |
| Crossover Rate          | 0.8    |
| Mutation Rate           | 0.01   |
| Max Depth               | 3      |

### Evaluation Metrics for GP

#### MAE on Training Data
| Parameter |               Value  |
|----------|-----------------------|
| Best     | 0.0009562653964029232 |
| Worst    | 0.06427222974196554   |
| Mean     | 0.012229803089428025  |
| Variance | 0.0002832592791702785 |

#### MAE on Test Data
| Parameter               | Value  |
|----------|-----------------------|
| Best     | 5.8217225537185135e-05 |
| Worst    | 0.0627240914890389     |
| Mean     | 0.012026128059825955   |
| Variance | 0.00026577231979841963 |

#### MSE on Training Data
| Parameter               | Value  |
|----------|-----------------------|
| Best     | 1.2676389649291442e-06 |
| Worst    | 0.004132187348331608   |
| Mean     | 0.00043388791281487717 |
| Variance | 9.388958524104617e-07  |

#### MSE on Test Data
| Parameter               | Value  |
|----------|-----------------------|
| Best     | 1.040942969541881e-07 |
| Worst    | 0.003934415403339194  |
| Mean     | 0.00041048467494230606 |
| Variance | 8.39005150587305e-07   |

### Best Solution Evolved by GP with Respect to Test Data

- **Best Value:** 1.1869640683635057
- **Best Equation:** $add(0.3332234961604388, cos(add(-0.4523309681912573, 1.0)))$

### ANN Architecture-Defining Parameters

| Parameter             | Value          |
|-----------------------|----------------|
| Model                 | Sequential     |
| Input Shape           | (2, )           |
| Regularization        | l2             |
| Hidden Layers         | 2              |
| Output Layer          | 1              |
| Activation Function   | relu           |
| Optimizer             | adam           |
| epochs                | 100            |
| Batch Size            | 32             |
| Loss Function         | MSE            |
| Metrics               | MAE            |

### Evaluation Metrics for ANN

#### MAE on Training Data
| Parameter         | Value              |
|-------------------|--------------------|
| Best     | 1.0354608297348022 |
| Worst    | 1.0354608297348022 |
| Mean     | 1.0354608297348022 |

#### MAE on Test Data
| Parameter         | Value              |
|-------------------|--------------------|
| Best     | 0.7849988341331482 |
| Worst    | 0.7849988341331482 |
| Mean     | 0.7849988341331482 |

#### MSE on Training Data
| Parameter         | Value              |
|-------------------|--------------------|
| Best     | 1.1907569180299624 |
| Worst    | 1.421831488609314  |
| Mean     | 1.3062942033196383 |
| Variance | 0.01334886429210793 |

#### MSE on Test Data
| Parameter         | Value              |
|-------------------|--------------------|
| Best     | 0.675733387318889  |
| Worst    | 0.9068063497543335 |
| Mean     | 0.7912698685366113 |
| Variance | 0.013348678492173086 |


## Objectives

-  Implement a more efficient GP algorithm to improve the results.
-  Utilize various types of ANN to find the best-performing model.
-  Use Mean Absolute Error (MAE) and Mean Squared Error (MSE) as evaluation metrics.
-  Perform an ensemble of 50 independent runs to show the best, worst, mean, and variance figures for MAE and MSE on both training and test data.
-  Report the best (with respect to the test data) solution evolved by GP and ANN.
-  Document all algorithm parameters, including default parameters used by the packages.

### Data Generation

The training and test data is generated using the following parameters:

Temperature values: [873, 973, 1073, 1173, 1273] K
Perturbed temperature and current values to introduce randomness
Various constants and parameters related to SOFC stack modeling

## GP Implementation

We use the DEAP library for GP implementation with the following features:

-  A primitive set that includes basic mathematical operations and trigonometric functions.
-  Evolutionary operators such as crossover and mutation.
-  A fitness function to evaluate the models based on MAE and MSE.
-  An ensemble approach with 50 independent runs to ensure robustness of the results.

## ANN Implementation

We use TensorFlow and Keras for ANN implementation with the following architecture:

-  Input layer with 2 features (temperature and current)
-  Two hidden layers with ReLU activation and L2 regularization
-  Output layer for regression
-  Training and evaluation on generated training and test data

## Results

We evaluate the models using both MAE and MSE on training and test data. The best, worst, mean, and variance figures are reported for an ensemble of 50 runs. Additionally, the best model (with respect to the test data) is documented along with the evolved equation from GP and the final equation from ANN.


## Insights on Machine Learning and ANN

### Genetic Programming (GP):

**Adaptability**: GP is highly adaptable for complex problem-solving where the form of the solution is not known beforehand.

**Symbolic Regression**: It can be particularly effective for symbolic regression tasks, where the goal is to find an algebraic expression that best fits the data.

**Interpretability**: The evolved equations are interpretable, which can be beneficial for understanding the underlying relationships in the data.


### Artificial Neural Networks (ANN):

**Flexibility**: ANNs are flexible and can model complex, non-linear relationships between inputs and outputs.

**Generalization**: They can generalize well to unseen data, as indicated by the close performance on training and testing data.

**Regularization**: Techniques like L2 regularization are used to prevent overfitting, ensuring the model performs well on both training and test data.

**Hyperparameter Tuning**: The architecture of the ANN (number of layers, number of neurons per layer, activation functions) and the training process (learning rate, batch size) can significantly impact performance.


## References

- [Energy 2009 paper, Uday Kumar Chakraborty](https://www.sciencedirect.com/science/article/abs/pii/S0360544209000449)

