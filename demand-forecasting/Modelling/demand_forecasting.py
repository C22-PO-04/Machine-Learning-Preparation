import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime, date, timedelta
import joblib
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


class DataPreprocessing:
    def __init__(self, scaler,  window_size=30, batch_size=8, buffer_size=100_000, stateful=False):
        self.scaler = scaler 
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.stateful = stateful

    def fit(self, data):
        self.scaler.fit(data[['sales', 'price']])
        
    def rescale(self, data):
        data = data.copy()
        scaled_data = self.scaler.transform(data[['sales', 'price']])
        data['scaled_sales'] = scaled_data[:,0]
        data['scaled_price'] = scaled_data[:,1]
        return data
    
    def create_windowed_dataset(self, data, with_label=True, rescale=True):
        if rescale:
            data = self.rescale(data)
        
        data = data[['scaled_sales', 'scaled_price', 'sales']].values
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.window(self.window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.window_size + 1))

        if not self.stateful:
            ds = ds.shuffle(self.buffer_size)

        if with_label:
            ds = ds.map(lambda w: ((w[:-1, :2], w[-1:, 1]), w[-1:, 2]))
        else:
            ds = ds.map(lambda w: (w[:-1, :2], w[-1:, 1]))

        ds = ds.batch(self.batch_size, drop_remainder=self.stateful)
        if not with_label:
            ds = ds.prefetch(1)

        return ds


class DiscountCalculator:
    def get_discounted_price(self, discount, days_ahead=14):
        base_prices, promos = self._parse(discount)
        future_price = []
        day = datetime.now()

        for _ in range(1, days_ahead+1):
            discounted_prices = base_prices 
            for promo in promos:
                self.is_in_range(day, promo['start_date'], promo['end_date'])
                if self.is_in_range(day, promo['start_date'], promo['end_date']):
                    for i in range(len(base_prices)):
                        discounted_prices[i] -= self.get_price_change(base_prices[i],
                            promo['discount'], 
                            promo['max_discount'])
                    future_price.append(sum(discounted_prices)/len(base_prices))
                else:
                    future_price.append(sum(base_prices)/len(base_prices))
            day = self.next_day(day)

        return future_price

    def get_price_change(self, base_price, discount, max_discount):
        price_change = base_price * discount
        return price_change if price_change < max_discount else max_discount

    def _parse(self, discount):
        return (
            discount['base_price'],
            discount['promo']
        )

    def is_in_range(self, day, start_date, end_date):
        return start_date <= day <= end_date 

    def next_day(self, day):
        return day + timedelta(days=1)


class ModelPipeline:
    def __init__(self, model, data_pipeline, discount_calculator):
        self.model = model
        self.data_pipeline = data_pipeline
        self.discount_calculator = discount_calculator
        
    def predict_next_week(self, data):
        return self.model.predict(data)[0][0]

    def forecast(self, data, prices):
        data = data.iloc[-(self.data_pipeline.batch_size + self.data_pipeline.window_size):]
        data = self.data_pipeline.rescale(data)
        data = data[['scaled_sales', 'scaled_price', 'sales']].copy()

        predictions = []
        for price in prices:
            data = data.append({'scaled_sales':0, 'scaled_price': self.data_pipeline.scaler.transform([[0, price]])[0, 1]}, ignore_index = True)
            current_week = self.data_pipeline.create_windowed_dataset(data, with_label=True, rescale=False)
            next_week_sales = self.predict_next_week(current_week)
            predictions.append(int(next_week_sales))
            data['sales'].iloc[-1] = next_week_sales

        return predictions

    def _parse(self, history):
        data = []
        for timestep in history:
            data.append([timestep.sales, timestep.price])
        return pd.DataFrame(data, columns=['sales', 'price'])

    def estimate(self, history, discount):
        data = self._parse(history)
        discounted_price = self.discount_calculator.get_discounted_price(discount)

        return self.forecast(data, discounted_price)


def get_demand_estimator():
    model = tf.keras.models.load_model("assets/model/demand_forecasting")
    scaler = joblib.load("assets/model/demand_forecasting/scaler.joblib")

    discount_calculator = DiscountCalculator()
    data_pipeline = DataPreprocessing(scaler)
    model_pipeline = ModelPipeline(model, data_pipeline, discount_calculator)

    return model_pipeline

demand_estimator = get_demand_estimator()

