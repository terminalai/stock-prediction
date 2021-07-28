# Stock Prediction with Deep Learning Models in TensorFlow
A MA2132 Journal Project

>_Lee Jia Jie, Prannaya Gupta_
>
>_NUS High School_
>
>**_20 September 2019_**

## Question
What would be an optimal architecture for a model to predict stock prices with as high accuracy as possible, given the constraints of either a FFNN, LSTM, GRU or CNN, and the number of 36 different models which may be trained to optimize its hyperparameters?

## Abstract
In this Mathematics Journal, we have predicted stocks by training neural networks on data from Yahoo Finance. This was done using the programming Language Python on the Google Colaboratory Platform and using the Machine Learning Models: LSTM, GRU, CNN and FFNN. First, we found the optimal model by deriving the data from Yahoo Finance into a `data/NFLX.csv` file and preprocessing pipeline. We then split the data and trained the models and, after finding LSTM as the better of the models, found the most optimal architecture for the LSTM model. We lastly unscaled and compared the prices against historical data and found the Opening Price for Netflix on tomorrow, the 21st of September, 2019 to be **$287.181**. Based on this, we concluded that a LSTM model with the architecture - CuDNNLSTM layer with 256 neurons, dropout layer with a 0.4 dropout rate, flatten layer, dense layer with 512 neurons, dropout layer with a 0.4 dropout rate and the final dense layer with 1 output neuron, yields the most accurate results. In the end we tested against Apple, NVIDIA and DBS to see if it recognised their patterns as well. In the end used it to predict day high and low and finally find the day high and low predictions for the next 10 days (21 September - 1 October 2019).

## Introduction
In this Mathematics Journal, we are going to predict stocks by training neural networks on data from Yahoo Finance.

I had just been studying machine learning when this Statistics Journal was assigned and thought that building a simple stock prediction system would be a good way to practise what I had learned and at the same time, complete the project. Furthermore, I could benefit from my father’s expertise as a stock trader, and I gained from him a basic overview of how the stock market worked.

For this project, we will be training and evaluating the models on five years’ worth of Netflix’s stock price data taken from Yahoo Finance.

We will use the Python programming language and the machine learning and data science libraries: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib` and `pandas_datareader` to automatically download the data. Finally, all the code will be written in an IPython notebook and executed on Google Colaboratory, which provides free hosted GPU runtime.

We are taking the loss of each model as the mean squared error and our aim is to obtain as low a value as possible; it shall basically be our measure of accuracy.

## Finding the Optimal Model
### The Data
The dataset consists of 7 columns, the open, high, low, and closing price (OHLC), volume, 10 days moving average and 30 days moving average. Here are the first five rows:

|No|High|Low|Open|Close|Volume|10ma|30ma|
|--|--|--|--|--|--|--|--|
|0|0.025150|0.026576|0.025368|0.029174|0.183252|0.020205|0.035485|
|1|0.025814|0.027007|0.028723|0.027508|0.155173|0.021367|0.034735|
|2|0.022676|0.024546|0.026326|0.025356|0.094994|0.022035|0.033892|
|3|0.022900|0.023622|0.025828|0.024439|0.116922|0.022263|0.032924|
|4|0.021947|0.023994|0.022937|0.025555|0.081407|0.022273|0.032094|
<p align="center">
  <b>Table 1</b>: Sample data of Netflix stock prices
</p>

The models will take in all the data from the past 100 days as input and predict the next day’s opening price.

### Training Different Models
We will be training 4 different types of machine learning models in this project. Their architectures can be seen in the code at the end of this report.

The first model is a basic Feed-Forward Neural Network (FFNN), containing only dense layers.

The second model uses Long Short-Term Memory (LSTM), which is designed for time-series prediction. We will use the CuDNNLSTM layer from tensorflow.keras.

The third model uses Gated Recurrent Unit layers (GRU). Like the LSTM, it is designed for such tasks as stock prediction but is simpler and more computationally efficient. We will use the CuDNNGRU layer from tensorflow.keras.

Finally, the fourth model is a Convolutional Neural Network (CNN). This model is not commonly used in stock prediction and time-series prediction in general, and is included mostly for variety and entertainment value.

After training the four models, the losses on the test set are shown below:

|Model|Past Loss|
|---|---|
|FFNN|0.0085|
|LSTM|0.0017|
|GRU|0.0019|
|CNN|0.0057|
<p align="center">
  <b>Table 2</b>: Model test scores
</p>

Considering the graph of validation loss, it can be seen that the LSTM performs best, followed by the GRU model. The standard FFNN performs the worst of all. In the next section, we will optimize the LSTM’s hyperparameters.

## Finding the Optimal Architecture for the LSTM

It is now time to fine-tune the LSTM model, and we do this by trying out multiple combinations of hyperparameters, such as the learning rate, decay, number of layers, layer sizes, etc. Possible values for each of these hyperparameters are stored in a list, and all their combinations are used to train separate models which will then be compared with the test set.

Due to time constraints, we are only able to tweak a few of these parameters, and the ones we have chosen are as follows:
- the number of neurons in the LSTM layers. There are 2 possible values chosen: 256 and 512.
- the number of LSTM layers. There are 3 possible values: 1, 2, or 3.
- the dropout rate of the layers. There are 3 possible values: 0, 0.2, or 0.4.
- the number of neurons in the fully-connected layers. There are 2 possible values, the same as that of the LSTM (256 and 512).

There are 2 × 3 × 3 × 2 = 36 possible combinations, and thus 36 models will be trained. 

After training, TensorBoard is used to compare the models’ performances. The results are as shown below (Each model’s name is in the format `LSTM-{LSTM neurons}-{LSTM layers}-{dropout rate}-{Dense neurons}-{timestamp}`):

```LSTM-256-1-0-256-1568463366           0.0005474354467450587
LSTM-256-1-0-512-1568463375           0.0004492358456927297
LSTM-256-1-0.2-256-1568463384         0.001225208372959648
LSTM-256-1-0.2-512-1568463393         0.0007193362199947895
LSTM-256-1-0.4-256-1568463402         0.0009116612504893805
LSTM-256-1-0.4-512-1568463412         0.0004236454558421803
LSTM-256-2-0-256-1568463423           0.0017564928405644263
LSTM-256-2-0-512-1568463437           0.0023046846885014984
LSTM-256-2-0.2-256-1568463452         0.006309656870058354
LSTM-256-2-0.2-512-1568463468         0.0009144440267632222
LSTM-256-2-0.4-256-1568463485         0.0016942301395294422
LSTM-256-2-0.4-512-1568463502         0.0018224813235814081
LSTM-256-3-0-256-1568463520           0.002481179139302934
LSTM-256-3-0-512-1568463542           0.0016386102739488705
LSTM-256-3-0.2-256-1568463565         0.001755849872407613
LSTM-256-3-0.2-512-1568463590         0.0013674528536605922
LSTM-256-3-0.4-256-1568463615         0.00296733299996156
LSTM-256-3-0.4-512-1568463641         0.0015114462153766961
LSTM-512-1-0-256-1568463669           0.0007310977883582168
LSTM-512-1-0-512-1568463690           0.0006145404237459469
LSTM-512-1-0.2-256-1568463712         0.0007105961558409035
LSTM-512-1-0.2-512-1568463734         0.0006120822207509156
LSTM-512-1-0.4-256-1568463756         0.000960952670464073
LSTM-512-1-0.4-512-1568463780         0.0004549121336929281
LSTM-512-2-0-256-1568463803           0.0008366446312078658
LSTM-512-2-0-512-1568463836           0.0007797060322927256
LSTM-512-2-0.2-256-1568463870         0.0011098333661827971
LSTM-512-2-0.2-512-1568463906         0.0009010062998105937
LSTM-512-2-0.4-256-1568463941         0.0019515789606991936
LSTM-512-2-0.4-512-1568463977         0.002409091345308458
LSTM-512-3-0-256-1568464014           0.003118468977182227
LSTM-512-3-0-512-1568464062           0.0009144910732763545
LSTM-512-3-0.2-256-1568464110         0.002377724928288337
LSTM-512-3-0.2-512-1568464160         0.004663190593504731
LSTM-512-3-0.4-256-1568464210         0.0014436753872466986
LSTM-512-3-0.4-512-1568464262         0.0012397152696982684
```
<p align="center">
  <b>Table 3</b>: Model Test Results
</p>

It is obvious that the model with 256 neurons in its LSTM layers, 1 LSTM layer, a 0.4 dropout rate and 512 neurons in its final Dense layer performed the best in the test set.

## Unscaling and Comparing the Prices

For the stock predictions to be useful, they must of course be unscaled to obtain the actual prices. We have thus implemented a function, unscale_price, which basically takes the same MinMaxScaler used to scale the data, obtains the x<sub>min</sub> and xmax stored in the class, and uses those values to unscale the data.

Now that the actual prices have been obtained, the predictions can be compared with the actual price to see the actual error of the model in terms of money.

Here are the first five rows of the actual price predictions:

|Date|Predicted|Actual|
|--|--|--|
|2015-03-27|58.167995|59.330002|
|2015-03-30|57.471416|59.715714|
|2015-03-31|57.224842|60.110001|
|2015-04-01|57.108532|59.642857|
|2015-04-02|56.827667|59.071430|

<p align="center">
  <b>Table 4</b>: Sample Predictions on Netflix data
</p>

As you can see, the model has an error of around a few dollars. The mean absolute error of all the predictions is calculated to be around $4.204.

Unfortunately, this is too high and definitely not practical in stock trading.

### Tomorrow’s Opening Price

Nevertheless, we used the LSTM model to predict the opening price of the next day (21/09/19), which was unscaled and yielded:
<p align="center">
  <b>$287.181</b>
</p>

## Conclusion and other improvements




