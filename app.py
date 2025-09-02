from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import numpy as np
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ---------------------------
# Load trained crop model
# ---------------------------
crop_model_path = os.path.join("models", "crop_model.pkl")
crop_model = joblib.load(crop_model_path)

# ---------------------------
# Mandi price dataset path
# ---------------------------
MANDI_DATA_PATH = "data/mandi_data.csv"

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Crop recommendation input
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N','P','K','temperature','humidity','ph','rainfall'])
        crop_prediction = crop_model.predict(input_data)[0]

        # Mandi price prediction input
        state = request.form['state']
        district = request.form['district']
        market = request.form['market']
        crop = crop_prediction

        # Load dataset
        df = pd.read_csv(MANDI_DATA_PATH)
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True)
        df = df.sort_values('Arrival_Date')

        # ---------------------------
        # Fallback logic for insufficient data
        # ---------------------------
        crop_df = df[(df['Commodity']==crop) & 
                     (df['State']==state) & 
                     (df['District']==district) & 
                     (df['Market']==market)][['Arrival_Date','Modal_x0020_Price']]

        if len(crop_df) < 2:
            crop_df = df[(df['Commodity']==crop) & (df['State']==state)][['Arrival_Date','Modal_x0020_Price']]
        if len(crop_df) < 2:
            crop_df = df[df['Commodity']==crop][['Arrival_Date','Modal_x0020_Price']]
        if len(crop_df) < 2:
            # Fallback: generate smooth dummy trend
            today = pd.to_datetime('today')
            last_price = 5000  # default reasonable price
            crop_df = pd.DataFrame({
                'ds': [today - pd.DateOffset(months=i) for i in range(12,0,-1)],
                'y': [last_price + i*50 for i in range(12)]
            })
        else:
            crop_df = crop_df.rename(columns={'Arrival_Date':'ds','Modal_x0020_Price':'y'}).reset_index(drop=True)

        # ---------------------------
        # Train Prophet model
        # ---------------------------
        model = Prophet()
        model.fit(crop_df)

        # Predict next 6 months
        future = model.make_future_dataframe(periods=6, freq='MS')
        forecast = model.predict(future)
        future_forecast = forecast.tail(6)[['ds','yhat']]

        predicted_prices = {f"Month {i+1}": round(price,2) 
                            for i,(date,price) in enumerate(zip(future_forecast['ds'], future_forecast['yhat']))}

        max_month_index = future_forecast['yhat'].idxmax()
        max_price = round(future_forecast.loc[max_month_index,'yhat'],2)
        max_month = future_forecast.loc[max_month_index,'ds'].strftime("%B %Y")

        # ---------------------------
        # Plot trend
        # ---------------------------
        plt.figure(figsize=(8,4))
        plt.plot(crop_df['ds'], crop_df['y'], marker='o', label='Past Price')
        plt.plot(future_forecast['ds'], future_forecast['yhat'], linestyle='--', label='Predicted Price')
        plt.xlabel('Date')
        plt.ylabel('Modal Price (₹/quintal)')
        plt.title(f'{crop} Price Trend ({state}, {district})')
        plt.legend()
        plt.tight_layout()
        plot_file = f'static/{crop}_price_trend.png'
        os.makedirs('static', exist_ok=True)
        plt.savefig(plot_file)
        plt.close()

        # ---------------------------
        # Calculate R² safely
        # ---------------------------
        if len(crop_df) < 2:
            r2_score = None
        else:
            y_true = crop_df['y']
            y_pred = forecast.loc[:len(crop_df)-1,'yhat']
            r2_score = round(np.corrcoef(y_true, y_pred)[0,1]**2, 2)

        return render_template('result.html',
                               crop=crop,
                               state=state,
                               district=district,
                               market=market,
                               predicted_prices=predicted_prices,
                               max_month=max_month,
                               max_price=max_price,
                               plot_file=plot_file,
                               r2_score=r2_score)

if __name__ == '__main__':
    app.run(debug=True)
