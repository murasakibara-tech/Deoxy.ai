from flask import Flask, render_template, request, g
import joblib
import plotly.express as px
import plotly.io as pio
import pandas as pd
import time

#The creation of a Flask Application
app = Flask(__name__)


#When the application starts, load the trained Random Forest Regressor model
model_full = joblib.load('DEOXY_rf_FULL_model.pkl')
model_basic = joblib.load('DEOXY_rf_BASIC_model.pkl')

# Recording will start when a request is ordered, this will be done to measure the response time
@app.before_request
def start_timer():
    g.start_time = time.perf_counter()

# This section will calculate how long the request took, then it would print out the results to the terminal
@app.after_request
def record_response_time(response):
    response_time_ms = (time.perf_counter() - g.start_time) * 1000
    print(f"{request.method} {request.path} took {response_time_ms:.2f}ms")
    response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
    return response
# Anything that appears on the homepage will be defined

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #Application will capture the values that the user used typing in the forms.
        temperature = float(request.form['Temperature'])
        salinity = float(request.form['Salinity'])
        depth = float(request.form['Depth'])
        latitude = float(request.form['Latitude'])
        longitude = float(request.form['Longitude'])
        year = float(request.form['Year'])
        month = float(request.form['Month'])
    except ValueError:
        #If whats written in the box isnt a number or perhaps blank then this message will be used instead of my dashboard just crashing
        return render_template('index.html', error="I request you enter only valid numbers in each and every field")
    

    #Was the advanced mode is turned on by the user AND did they filled out the nutirent fields.
    advanced = request.form.get('Advanced') == 'yes'
    nitrate = request.form.get('Nitrate', '').strip()
    phosphate = request.form.get('Phosphate', '').strip()
    silicate = request.form.get('Silicate', '').strip()

    use_advanced = advanced and nitrate and phosphate and silicate

    if use_advanced:
        try:
            nitrate = float(nitrate); phosphate = float(phosphate); silicate = float(silicate)
        except ValueError:
            return render_template('index.html', error = "I request you only enter valid numbers inside the nutrient fields, or switch off Advanced mode")


        #My model is expecting that the features will come in the same order it was previoussly trained on.
        #This is the full version, so the full feature order would need to match the training done: temperature, salinity, depth, nitrate, phosphate, silicate, latitude, longitude, year, month
        features = [[temperature, salinity, depth, nitrate, phosphate, silicate, latitude, longitude, year, month]]
    #Next my model will be asked for its prediction
        prediction = model_full.predict(features)[0]
        mode_used = "Advanced (with nutrients)"
    else:
        #This is the basic feature order: temperature, salinity, depth, latitude, longitude, year, month
        features = [[temperature, salinity, depth, latitude, longitude, year, month ]]
        prediction = model_basic.predict(features)[0]
        mode_used = "Basic (no nutrients)"

    # Classifcation for the different amount of predicted oxygen into an established risk band by utulising my projects existing thresholds
    if prediction <60:
        risk = "Hypoxic (dead zone)"
    elif prediction <120:
        risk = "Low oxygen (at-risk)"
    else:
        risk = "Healthy"

    model_names = ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

    r2_values = [0.80, 0.80, 0.95, 0.97, 0.95]

    fig = px.bar(x=r2_values, y=model_names, orientation='h',
                 labels={'x': 'R² Score', 'y': 'Model'},
                 title='Model Comparison (R²)', range_x=[0,1])
    
    chart_html = pio.to_html(fig, full_html=False)

    #Now the scatter chart will be read through my saved HTML fille.
    with open('actual_vs_predicted.html','r') as f:
        scatter = f.read()
        
    #then the results would be sent back to the page as well as the risk band, rounded up to 2 decimal places 
    return render_template('index.html',
                    prediction=round(prediction, 2),
                    risk=risk, chart=chart_html, scatter = scatter)


@app.route('/')
def home():
    #In order to chart the ModelResult-style info that i had in colab i need to first load the cleaned dataset
    model_names = ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

    r2_values = [0.80, 0.80, 0.95, 0.99, 0.97]

    fig = px.bar(x=r2_values, y=model_names, orientation='h',
                 labels={'x': 'R² Score', 'y': 'Model'},
                 title='Model Comparison (R²)', range_x=[0,1])
    
    chart_html = pio.to_html(fig, full_html=False)

    #Now the scatter chart will be read through my saved HTML fille.
    with open('actual_vs_predicted.html','r') as f:
        scatter = f.read()


    return render_template('index.html', chart=chart_html, scatter=scatter)



#Running the application
if __name__ == '__main__':
    app.run(debug=True)
