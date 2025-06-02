from flask import request, jsonify, render_template, flash
import numpy as np
import pandas as pd
import logging
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import MeanSquaredError
from config import CONFIG
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from glob import glob
import os
import json
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


sale_periods = {
    ('Azadpur', 'Makhmali', 'Fancy'): {'start': '05-20', 'end': '06-30', 'years': list(range(2015, 2026))},
    ('Azadpur', 'Makhmali', 'Special'): {'start': '05-20', 'end': '06-30', 'years': list(range(2015, 2026))},
    ('Azadpur', 'Makhmali', 'Super'): {'start': '05-20', 'end': '06-30', 'years': list(range(2015, 2026))},
    ('Azadpur', 'Misri', 'Fancy'): {'start': '06-10', 'end': '07-15', 'years': list(range(2012, 2026))},
    ('Azadpur', 'Misri', 'Special'): {'start': '06-10', 'end': '07-15', 'years': list(range(2012, 2026))},
    ('Azadpur', 'Misri', 'Super'): {'start': '06-10', 'end': '07-15', 'years': list(range(2012, 2026))},
    ('Ganderbal', 'Cherry', 'Large'): {'start': '05-13', 'end': '06-28', 'years': list(range(2019, 2026))},
    ('Ganderbal', 'Cherry', 'Medium'): {'start': '05-13', 'end': '06-28', 'years': list(range(2019, 2026))},
    ('Ganderbal', 'Cherry', 'Small'): {'start': '05-13', 'end': '06-28', 'years': list(range(2019, 2026))},
    ('Narwal', 'American', 'Unknown'): {'start': '09-01', 'end': '12-30', 'years': list(range(2011, 2026))},
    ('Narwal', 'Cherry', 'Large'): {'start': '05-15', 'end': '07-10', 'years': list(range(2016, 2026))},
    ('Narwal', 'Cherry', 'Medium'): {'start': '05-15', 'end': '07-10', 'years': list(range(2016, 2026))},
    ('Narwal', 'Cherry', 'Small'): {'start': '05-15', 'end': '07-10', 'years': list(range(2016, 2026))},
    ('Narwal', 'Condition', 'Unknown'): {'start': '07-01', 'end': '08-31', 'years': list(range(2019, 2026))},
    ('Narwal', 'Delicious', 'Unknown'): {'start': '01-01', 'end': '12-31', 'years': list(range(2011, 2026))},
    ('Narwal', 'Hazratbali', 'Unknown'): {'start': '07-01', 'end': '08-30', 'years': list(range(2011, 2026))},
    ('Narwal', 'Razakwadi', 'Unknown'): {'start': '08-01', 'end': '08-30', 'years': list(range(2013, 2026))},
    ('Parimpore', 'Cherry', 'Large'): {'start': '05-10', 'end': '06-30', 'years': list(range(2019, 2026))},
    ('Parimpore', 'Cherry', 'Medium'): {'start': '05-10', 'end': '06-30', 'years': list(range(2019, 2026))},
    ('Parimpore', 'Cherry', 'Small'): {'start': '05-10', 'end': '06-30', 'years': list(range(2019, 2026))},
    ('Shopian', 'American', 'A'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'American', 'B'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'Cherry', 'Large'): {'start': '05-20', 'end': '07-10', 'years': list(range(2019, 2026))},
    ('Shopian', 'Cherry', 'Medium'): {'start': '05-20', 'end': '07-10', 'years': list(range(2019, 2026))},
    ('Shopian', 'Cherry', 'Small'): {'start': '05-20', 'end': '07-10', 'years': list(range(2019, 2026))},
    ('Shopian', 'Delicious', 'A'): {'start': '09-15', 'end': '12-31', 'years': list(range(2017, 2026))},
    ('Shopian', 'Delicious', 'B'): {'start': '09-15', 'end': '12-31', 'years': list(range(2017, 2026))},
    ('Shopian', 'Kullu delicious', 'A'): {'start': '09-01', 'end': '12-15', 'years': list(range(2017, 2026))},
    ('Shopian', 'Kullu delicious', 'B'): {'start': '09-01', 'end': '12-15', 'years': list(range(2017, 2026))},
    ('Shopian', 'Maharaji', 'A'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Shopian', 'Maharaji', 'B'): {'start': '10-01', 'end': '11-30', 'years': list(range(2017, 2026))},
    ('Sopore', 'American', 'A'): {'start': '01-01', 'end': '12-31', 'years': list(range(2015, 2026))},
    ('Sopore', 'American', 'B'): {'start': '01-01', 'end': '12-31', 'years': list(range(2015, 2026))},
    ('Sopore', 'Delicious', 'A'): {'start': '01-01', 'end': '12-31', 'years': list(range(2015, 2026))},
    ('Sopore', 'Delicious', 'B'): {'start': '01-01', 'end': '12-31', 'years': list(range(2015, 2026))},
    ('Sopore', 'Maharaji', 'A'): {'start': '11-01', 'end': '12-30', 'years': list(range(2015, 2026))},
    ('Sopore', 'Maharaji', 'B'): {'start': '11-01', 'end': '12-30', 'years': list(range(2015, 2026))}
}


def create_forecast_plot(forecast_dates, future_predictions):
    trace = go.Scatter(x=forecast_dates, y=future_predictions, mode='lines+markers', name='Forecast')
    layout = go.Layout(title='Forecasted Prices', xaxis_title='Date', yaxis_title='Price (per kg)',
                       template='plotly_white', margin=dict(l=30, r=30, t=50, b=30))
    return pio.to_html(go.Figure(data=[trace], layout=layout), full_html=False)

def align_forecast_dates_to_previous_year(df, forecast_days, target_year):
    df = df.copy()
    df['MonthDay'] = df['Date'].dt.strftime('%m-%d')
    unique_md = sorted(df['MonthDay'].unique())
    if len(unique_md) < forecast_days:
        raise ValueError("Not enough date variety in past data for forecast window")
    return [f"{target_year}-{md}" for md in unique_md[:forecast_days]]

def create_dashboard_plot(df):
    import plotly.graph_objs as go
    import plotly.io as pio
    import json

    # Filter only actual sales
    df = df[df['Mask'] == 1].copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Create a Plotly scatter line chart without connecting missing dates
    trace = go.Scatter(
        x=df['Date'],
        y=df['Price (₹/kg)'],
        mode='lines+markers',
        line=dict(color='orange'),
        marker=dict(size=6),
        name='Actual Sales',
        connectgaps=False  # This is the key fix
    )

    layout = go.Layout(
        title='Recent Price Trends (Only Actual Sales)',
        xaxis_title='Date',
        yaxis_title='Price (₹/kg)',
        template='plotly_white',
        margin=dict(l=40, r=30, t=50, b=40)
    )

    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_html(fig, full_html=False)

def parse_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df[df['Mask'] == 1]
        df['Date'] = pd.to_datetime(df['Date'])
        df['Price'] = df['Avg Price (per kg)']
        df.rename(columns={'Avg Price (per kg)': 'Price (₹/kg)'}, inplace=True)

        parts = file_path.split(os.sep)
        market = parts[-2]
        file_name = os.path.basename(file_path).replace('_dataset.csv', '')
        tokens = file_name.split('_')

        if len(tokens) == 2:
            variety, grade = tokens
        elif len(tokens) == 3:
            variety = f"{tokens[0]} {tokens[1]}"
            grade = tokens[2]
        else:
            return None

        fruit = 'Cherry' if 'cherry' in file_path.lower() else 'Apple' if 'apple' in file_path.lower() else 'Unknown'
        df['Market'], df['Fruit'], df['Variety'], df['Grade'] = market, fruit, variety, grade

        return df[['Date', 'Market', 'Fruit', 'Variety', 'Grade', 'Price (₹/kg)', 'Price']]
    except Exception as e:
        logging.warning(f"Skipping file {file_path} due to error: {e}")
        return None

def get_config_options(selected_market, selected_fruit=None, selected_variety=None):
    fruits = sorted(CONFIG[selected_market].keys()) if selected_market in CONFIG else []
    varieties = sorted(CONFIG[selected_market][selected_fruit].keys()) if selected_market in CONFIG and selected_fruit in CONFIG[selected_market] else []
    grades = sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys()) if selected_market in CONFIG and selected_fruit in CONFIG[selected_market] and selected_variety in CONFIG[selected_market][selected_fruit] else []
    return fruits, varieties, grades

def setup_routes(app):
    @app.route('/')
    def home():
        try:
            markets = sorted(CONFIG.keys())
            selected_market = request.args.get('market', markets[0] if markets else '')
            fruits = sorted(CONFIG[selected_market].keys()) if selected_market in CONFIG else []
            selected_fruit = request.args.get('fruit', fruits[0] if fruits else '')
            varieties = sorted(CONFIG[selected_market][selected_fruit].keys()) if selected_fruit in CONFIG[selected_market] else []
            selected_variety = request.args.get('variety', varieties[0] if varieties else '')
            grades = sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys()) if selected_variety in CONFIG[selected_market][selected_fruit] else []
            selected_grade = request.args.get('grade', grades[0] if grades else '')

            sale_periods_json = { "|".join(k): v for k, v in sale_periods.items() }
            return render_template('predict.html',
                               config=CONFIG,
                               sale_periods=sale_periods_json,  # 🛠 Add this line
                               markets=markets,
                               fruits=fruits,
                               varieties=varieties,
                               grades=grades,
                               selected_market=selected_market,
                               selected_fruit=selected_fruit,
                               selected_variety=selected_variety,
                               selected_grade=selected_grade)
        except Exception as e:
            logging.error(f"Error rendering template: {e}")
            return "Template not found", 404
    
    @app.route('/predict_future', methods=['POST'])
    def predict_future():
        try:
            selected_market = request.form.get('market')
            selected_fruit = request.form.get('fruit')
            selected_variety = request.form.get('variety')
            selected_grade = request.form.get('grade')
            forecast_option = request.form.get('forecast_option')  # 'week' or 'fortnight'
            start_date_str = request.form.get('start_date')

            if not start_date_str or not forecast_option:
                return jsonify({'error': 'Start date and forecast option are required.'}), 400

            start_date = pd.to_datetime(start_date_str)
            
            # Sale periods dictionary lookup
            sale_key = (selected_market, selected_variety, selected_grade)
            sale_info = sale_periods.get(sale_key)
            if not sale_info:
                return jsonify({'error': f'No sale period defined for {selected_market}, {selected_variety}, {selected_grade}.'}), 400

            current_year = datetime.now().year
            if current_year not in sale_info['years']:
                return jsonify({'error': f'No sale period defined for {selected_market}, {selected_variety}, {selected_grade} in {current_year}.'}), 400

            sale_start_date = pd.to_datetime(f"{current_year}-{sale_info['start']}")
            sale_end_date = pd.to_datetime(f"{current_year}-{sale_info['end']}")

            if start_date < sale_start_date or start_date > datetime.now():
                return jsonify({'error': f'Start date {start_date.strftime("%Y-%m-%d")} is outside allowed range ({sale_start_date.strftime("%Y-%m-%d")} to today).'}), 400

            # Load data and model
            config_entry = CONFIG[selected_market][selected_fruit][selected_variety][selected_grade]
            model_path = config_entry['model']
            data_path = config_entry['dataset']
            
            df = pd.read_csv(data_path)
            df = df[df['Mask'] == 1]
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(by='Date', inplace=True)
            prices = df['Avg Price (per kg)'].values.reshape(-1, 1)
            scaler = MinMaxScaler().fit(prices)
            model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
            time_steps = model.input_shape[1]
            
            # Generate full sale period forecast
            total_forecast_days = (sale_end_date - sale_start_date).days + 1
            input_sequence = scaler.transform(prices[-time_steps:]).reshape(1, time_steps, 1)
            future_predictions = []
            for _ in range(total_forecast_days):
                prediction = model.predict(input_sequence, verbose=0)
                predicted_price = scaler.inverse_transform(prediction)
                future_predictions.append(float(predicted_price[0][0]))
                input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
            
            forecast_dates_all = pd.date_range(start=sale_start_date, periods=total_forecast_days).to_list()
            # Filter forecasts starting from start_date
            filtered = [(d, p) for d, p in zip(forecast_dates_all, future_predictions) if d >= start_date]

            forecast_length = 7 if forecast_option == 'week' else 14
            filtered = filtered[:forecast_length]
            if not filtered:
                return jsonify({'error': f'No forecasts available starting from {start_date}.'}), 400
            
            forecast_dates = [d.strftime('%Y-%m-%d') for d, _ in filtered]
            filtered_prices = [p for _, p in filtered]
            
            forecast_plot = create_forecast_plot(forecast_dates, filtered_prices)
            predicted_prices = list(zip(forecast_dates, filtered_prices))

            fruits, varieties, grades = get_config_options(selected_market, selected_fruit, selected_variety)
            sale_periods_json = { "|".join(k): v for k, v in sale_periods.items() }
            return render_template('predict.html',
                       config=CONFIG,
                       sale_periods=sale_periods_json,  # Convert tuple keys to strings
                       markets=sorted(CONFIG.keys()),
                       fruits=fruits,
                       varieties=varieties,
                       grades=grades,
                       selected_market=selected_market,
                       selected_fruit=selected_fruit,
                       selected_variety=selected_variety,
                       selected_grade=selected_grade,
                       predicted_prices=predicted_prices,
                       trend_plot=forecast_plot,
                       start_date=start_date.strftime('%Y-%m-%d'))
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return jsonify({'error': 'Prediction failed.'}), 500


    @app.route('/dashboard')
    def dashboard():
        try:
            # Reconstruct CONFIG for dashboard dropdown rendering (preserve original for prediction)
            dashboard_config = {}
            for market, fruits in CONFIG.items():
                if market == "Pulwama":
                    for fruit, submarkets in fruits.items():
                        for submarket, grades in submarkets.items():
                            new_market_key = f"{submarket} Pulwama"
                            if new_market_key not in dashboard_config:
                                dashboard_config[new_market_key] = {}
                            dashboard_config[new_market_key][fruit] = grades
                else:
                    dashboard_config[market] = fruits

            markets = sorted(dashboard_config.keys())
            selected_market = request.args.get('market') or markets[0] if markets else ''

            # Map Pulwama submarkets back
            adjusted_market = "Pulwama"
            location_key = None
            if selected_market.startswith("Pachhar"):
                location_key = "Pachhar"
            elif selected_market.startswith("Prichoo"):
                location_key = "Prichoo"
            else:
                adjusted_market = selected_market

            fruits = sorted(dashboard_config[selected_market].keys()) if selected_market in dashboard_config else []
            selected_fruit = request.args.get('fruit') or (fruits[0] if fruits else '')
            varieties = []
            if location_key:
                varieties = sorted(CONFIG[adjusted_market][selected_fruit][location_key].keys()) if selected_fruit in CONFIG[adjusted_market] and location_key in CONFIG[adjusted_market][selected_fruit] else []
            else:
                varieties = sorted(CONFIG[selected_market][selected_fruit].keys()) if selected_fruit in CONFIG[selected_market] else []

            selected_variety = request.args.get('variety') or (varieties[0] if varieties else '')
            grades = []
            if location_key:
                grades = sorted(CONFIG[adjusted_market][selected_fruit][location_key][selected_variety].keys()) if selected_variety in CONFIG[adjusted_market][selected_fruit][location_key] else []
            else:
                grades = sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys()) if selected_variety in CONFIG[selected_market][selected_fruit] else []

            selected_grade = request.args.get('grade') or (grades[0] if grades else '')

            cards = [
                {'title': 'Selected Market', 'value': selected_market or 'N/A'},
                {'title': 'Selected Fruit', 'value': selected_fruit or 'N/A'},
                {'title': 'Selected Variety', 'value': selected_variety or 'N/A'},
                {'title': 'Selected Grade', 'value': selected_grade or 'N/A'}
            ]

            data = []
            plot_json = '[]'

            try:
                if location_key:
                    config_entry = CONFIG[adjusted_market][selected_fruit][location_key][selected_variety][selected_grade]
                else:
                    config_entry = CONFIG[selected_market][selected_fruit][selected_variety][selected_grade]

                data_path = config_entry['dataset']
                df = pd.read_csv(data_path)
                df = df[df['Mask'] == 1]
                df['Date'] = pd.to_datetime(df['Date'])
                df.sort_values(by='Date', inplace=True)
                df['Price'] = df['Avg Price (per kg)']
                df.rename(columns={'Avg Price (per kg)': 'Price (₹/kg)'}, inplace=True)

                data = df.tail(150).to_dict(orient='records')
                if df.empty or 'Price (₹/kg)' not in df.columns:
                    flash("No data available for the selected combination.", "warning")
                    return render_template("dashboard.html", config=dashboard_config, data=[], plot_data="", selected_market=selected_market, selected_fruit=selected_fruit, selected_variety=selected_variety, selected_grade=selected_grade, cards=cards)
                plot_json = create_dashboard_plot(df)

            except Exception as e:
                logging.warning(f"No data available for the selected combination: {e}")
                flash("No data available for the selected combination.", "warning")

            plot_img = create_dashboard_plot(df.tail(100))
            return render_template("dashboard.html", config=dashboard_config, data=df.tail(150).to_dict(orient='records'), plot_data=plot_img, selected_market=selected_market, selected_fruit=selected_fruit, selected_variety=selected_variety, selected_grade=selected_grade, cards=cards)

        except Exception as e:
            logging.error(f"Dashboard error: {str(e)}")
            return render_template("dashboard.html", config=CONFIG, data=[], plot_data='[]', selected_market='', selected_fruit='', selected_variety='', selected_grade='', cards=[])
