from flask import Flask, render_template, request, redirect, url_for, session
from flask_dance.contrib.google import make_google_blueprint, google
import os
import pickle
import numpy as np
import pandas as pd
import re

# -------------------- App Configuration --------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")  # Load secret key from environment

# Allow OAuth over HTTP (for development only)
# os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# -------------------- Google OAuth Setup --------------------
google_bp = make_google_blueprint(
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"
    ],
    redirect_url="/google_login"
)
app.register_blueprint(google_bp, url_prefix="/login")

# -------------------- Session Handling --------------------
@app.before_request
def clear_session_on_restart():
    """Clear session once when the server restarts."""
    if not hasattr(app, 'session_cleared'):
        session.clear()
        app.session_cleared = True

# -------------------- Routes --------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Manual login route."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Simple authentication (no database used)
        session['user'] = {'username': username}
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/google_login')
def google_login():
    """Login via Google OAuth."""
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    user_info = resp.json()
    session['user'] = user_info
    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    """Logout and clear session."""
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/')
def index():
    """Home page - only accessible if logged in."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', cities=available_cities, locations=available_locations)

# -------------------- Load Model --------------------
with open('model/model_improved.pkl', 'rb') as f:
    model_data = pickle.load(f)
    MODEL = model_data['model']
    SCALER = model_data['scaler']
    NUMERIC_FEATURES = model_data['numeric_features']
    LOCATION_PRICE_IDX = model_data['location_price_idx']
    CITY_PRICE_IDX = model_data['city_price_idx']
    COLUMNS = model_data['columns']

available_cities = sorted(list(CITY_PRICE_IDX.keys()))
available_locations = sorted(list(LOCATION_PRICE_IDX.keys()))

# -------------------- Jinja Filter --------------------
def indian_format(value):
    """Format numbers in Indian currency style."""
    try:
        value = float(str(value).replace(",", ""))
        s = f"{int(value):d}"
        if len(s) > 3:
            last3 = s[-3:]
            rest = s[:-3]
            rest = re.sub(r"(\d)(?=(\d{2})+(?!\d))", r"\1,", rest)
            formatted = rest + "," + last3
        else:
            formatted = s
        decimals = f"{value:.2f}".split(".")[1]
        return formatted + "." + decimals
    except Exception:
        return value

app.jinja_env.filters['indian_format'] = indian_format

# -------------------- Prediction Route --------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Handle property detail submission and return predictions."""
    try:
        # Get form data
        city = request.form['city'].strip()
        location = request.form['location'].strip()
        area = float(request.form['area'])
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])

        # Validate city and location
        if location not in LOCATION_PRICE_IDX:
            raise ValueError(f"Location '{location}' not found in data.")
        if city not in CITY_PRICE_IDX:
            raise ValueError(f"City '{city}' not found in data.")

        # Create input DataFrame
        input_data = pd.DataFrame({
            'Location': [location],
            'City': [city],
            'Area': [area],
            'BHK': [bhk],
            'Bathroom': [bath]
        })

        # Feature engineering
        input_data['room_density'] = (input_data['BHK'] / input_data['Area']).clip(0, 1)
        input_data['bath_bed_ratio'] = (input_data['Bathroom'] / input_data['BHK']).clip(0, 4)

        # Map indices
        input_data['location_price_index'] = input_data['Location'].map(LOCATION_PRICE_IDX)
        input_data['city_price_index'] = input_data['City'].map(CITY_PRICE_IDX)

        # One-hot encode
        input_encoded = pd.get_dummies(input_data, columns=['Location', 'City'])

        # Add missing columns
        for col in COLUMNS:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Match column order
        input_encoded = input_encoded[COLUMNS]

        # Scale numerical features
        input_encoded[NUMERIC_FEATURES] = SCALER.transform(input_encoded[NUMERIC_FEATURES])

        # Predict price
        prediction = MODEL.predict(input_encoded)[0]

        # Growth rate calculation
        location_growth = LOCATION_PRICE_IDX.get(location, 0.10)
        city_growth = CITY_PRICE_IDX.get(city, 0.08)
        growth_rate = (0.7 * location_growth + 0.3 * city_growth)
        growth_rate = max(0.05, min(0.20, growth_rate))

        fv_1y = prediction * (1 + growth_rate)
        fv_3y = prediction * (1 + growth_rate) ** 3
        fv_5y = prediction * (1 + growth_rate) ** 5

        return render_template(
            "result.html",
            price=prediction,
            fv_1y=fv_1y,
            fv_3y=fv_3y,
            fv_5y=fv_5y
        )

    except Exception as e:
        return render_template("result.html", price=None, error=str(e))

# -------------------- Run App --------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
