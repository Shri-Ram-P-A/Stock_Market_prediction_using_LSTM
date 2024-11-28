### README: LSTM-Based Stock Price Prediction

---

#### Overview
This project utilizes Long Short-Term Memory (LSTM) networks to predict stock prices based on historical data. The implementation is focused on stock data, leveraging Python, TensorFlow, and supporting libraries for preprocessing, modeling, and visualization.

---

#### Features
1. **Data Preprocessing**:
   - Normalizes stock price data using MinMaxScaler to scale values between 0 and 1.
   - Splits the dataset into training (75%) and testing (25%) subsets.

2. **Dataset Preparation**:
   - Converts time-series data into a supervised learning format using a sliding window approach.

3. **LSTM Model**:
   - A stacked LSTM architecture is implemented with three LSTM layers and a dense output layer.
   - Optimized using the Adam optimizer and trained to minimize mean squared error (MSE).

4. **Prediction**:
   - Generates predictions on both training and testing data.
   - Predicts future stock prices for the next 30 days based on the latest historical data.

5. **Visualization**:
   - Visualizes the original stock price data along with training and testing predictions.
   - Projects future stock price predictions and visualizes them alongside recent historical data.

---

#### Requirements

The project requires the following libraries:

```plaintext
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

---

#### Installation

1. Clone the repository or download the source code.
2. Install the required Python libraries using pip:
   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow
   ```
3. Ensure the file `Stock.csv` is present in the working directory. The file should contain historical stock price data with a `Close` column.

---

#### Usage

1. **Run the Script**:
   - Execute the Python script in any IDE or terminal. It will load the stock price data, train the LSTM model, and predict future prices.

2. **Outputs**:
   - Plots of original vs. predicted prices for training and testing data.
   - A projection of the next 30 days of stock prices.

3. **Customizations**:
   - Modify the `time_step` parameter in the `create_dataset` function to change the input sequence length.
   - Adjust the number of epochs and batch size during model training to optimize performance.

---

#### File Descriptions

1. **`Stock.csv`**: The input file containing historical stock price data. Ensure it has a `Close` column for closing prices.
2. **Script**: Implements data preprocessing, model training, prediction, and visualization.

---

#### Example Plots

1. **Training and Testing Predictions**:
   Visualizes how well the model fits the training and testing datasets.

2. **Future Predictions**:
   Plots the model's predicted stock prices for the next 30 days alongside historical data.

---

#### Future Enhancements
- Incorporate more features like `Open`, `High`, `Low`, and `Volume` for a richer model.
- Experiment with other architectures like GRU or hybrid models.
- Add hyperparameter tuning for improved model accuracy.

---

Enjoy exploring stock price trends and predictions with this LSTM-based model!