import pandas as pd
import numpy as np
from datetime import datetime, timedelta

states = ['Punjab']
districts = ['Patiala']
markets = ['SampleMarket1']
crops = ['Rice', 'Wheat', 'Maize']

start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)

dates = pd.date_range(start_date, end_date, freq='MS')  # monthly start

data = []

for crop in crops:
    for date in dates:
        for state in states:
            for district in districts:
                for market in markets:
                    base_price = {
                        'Rice': 2000,
                        'Wheat': 1800,
                        'Maize': 1500
                    }[crop]
                    
                    # seasonal fluctuation ±10%
                    price = base_price * (1 + np.random.uniform(-0.1, 0.1))
                    min_price = round(price * 0.9, 2)
                    max_price = round(price * 1.1, 2)
                    modal_price = round(price, 2)
                    
                    row = [
                        state, district, market, crop, crop, 'FAQ',
                        date.strftime('%d/%m/%Y'), min_price, max_price, modal_price
                    ]
                    data.append(row)

columns = ['State','District','Market','Commodity','Variety','Grade','Arrival_Date',
           'Min_x0020_Price','Max_x0020_Price','Modal_x0020_Price']

df = pd.DataFrame(data, columns=columns)

df.to_csv('data/mandi_data.csv', index=False)
print("Dummy 5-year dataset generated: data/mandi_data.csv")
