import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self, window_size, batch_size, buffer_size, stateful):
        self.scaler = StandardScaler()
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.stateful = stateful
#         scaled_data = scaler.fit_transform(data[['sales', 'price']])

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

class ModelPipeline:
    def __init__(self, model, data_pipeline):
        self.model = model
        self.data_pipeline = data_pipeline
        
    def predict_next_week(self, model, data):
        return self.model.predict(data)[0][0]

    def forecast(self, data, prices):
        data = self.data_pipeline.rescale(data)
        data = data[['scaled_sales', 'scaled_price', 'sales']].copy()
        predictions = []
        for price in prices:
            data = data.append({'scaled_sales':0, 'scaled_price': self.data_pipeline.scaler.transform([[0, price]])[0, 1]}, ignore_index = True)
            current_week = self.data_pipeline.create_windowed_dataset(data.iloc[-(self.data_pipeline.batch_size + self.data_pipeline.window_size):], with_label=True, rescale=False)
            next_week_sales = self.predict_next_week(self.model, current_week)
            predictions.append(next_week_sales)
            data['sales'].iloc[-1] = next_week_sales

        return predictions
