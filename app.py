from flask import Flask, render_template, send_file, request, redirect, url_for, flash
import os
from flask_cors import CORS, cross_origin
from predictFromModelClassification import predictionClassification
from predictFromModelRegression import predictionRegression

app=Flask(__name__)
CORS(app)
app.secret_key = "any random string"


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")

@app.route("/fire", methods=['GET'])
@cross_origin()
def fire():
    return render_template('fire.html')

@app.route("/temperature", methods=['GET'])
@cross_origin()
def temperature():
    return render_template('temperature.html')

@app.route("/predict-fire", methods=['POST'])
@cross_origin()
def predict_fire():
    try:
        date=request.form['date']
        temperature=int(request.form['temperature'])
        rh=int(request.form['rh'])
        ws=int(request.form['ws'])
        rain=float(request.form['rain'])
        ffmc=float(request.form['ffmc'])
        dmc=float(request.form['dmc'])
        dc=float(request.form['dc'])
        isi=float(request.form['isi'])
        bui=float(request.form['bui'])
        fwi=float(request.form['fwi'])
        region=request.form['region']

        date=date.split('-')
        day, month, year = int(date[2]), int(date[1]), int(date[0])

        data={
            "day" : day,
            "month" : month,
            "year" : year,
            "Temperature" : temperature,
            "RH" : rh,
            "Ws" : ws,
            "Rain" : rain,
            "FFMC" : ffmc,
            "DMC" : dmc,
            "DC" : dc,
            "ISI" : isi,
            "BUI" : bui,
            "FWI" : fwi,
            "Region" : region
        }

        pred = predictionClassification(data)
        output = pred.predictionFromModelClassification()
        flash(f"The possibility is: {output}", "success")

        return redirect(url_for('fire'))

    except Exception :
        flash('Something went wrong', 'danger')
        return redirect(url_for('fire'))


@app.route("/predict-fire-dataset", methods=['POST'])
@cross_origin()
def predict_fire_dataset():
    try:
        filePath=request.files['filePath']
        fileName='Prediction_Classification.csv'

        if not os.path.isdir('Prediction_Files'):
            os.mkdir('Prediction_Files')
        path='Prediction_Files'+'/'+fileName
        filePath.save(path)

        pred = predictionClassification(path)
        output_folder=pred.predictionFromModelClassification()
        os.remove(path)
        return send_file(output_folder, as_attachment=True)
    except Exception:
        flash('Something went wrong', 'danger')
        return redirect(url_for('fire'))


@app.route("/predict-temperature", methods=['POST'])
@cross_origin()
def predict_temperature():
    try:
        date=request.form['date']
        classes=request.form['classes']
        rh=int(request.form['rh'])
        ws=int(request.form['ws'])
        rain=float(request.form['rain'])
        ffmc=float(request.form['ffmc'])
        dmc=float(request.form['dmc'])
        dc=float(request.form['dc'])
        isi=float(request.form['isi'])
        bui=float(request.form['bui'])
        fwi=float(request.form['fwi'])
        region=request.form['region']

        date=date.split('-')
        day, month, year = int(date[2]), int(date[1]), int(date[0])

        data={
            "day" : day,
            "month" : month,
            "year" : year,
            "Classes" : classes,
            "RH" : rh,
            "Ws" : ws,
            "Rain" : rain,
            "FFMC" : ffmc,
            "DMC" : dmc,
            "DC" : dc,
            "ISI" : isi,
            "BUI" : bui,
            "FWI" : fwi,
            "Region" : region
        }

        pred = predictionRegression(data)
        output = pred.predictionFromModelRegression()
        flash(f"The predicted temperature is: {output} degree Celsius", "success")

        return redirect(url_for('temperature'))

    except Exception :
        flash('Something went wrong', 'danger')
        return redirect(url_for('temperature'))


@app.route("/predict-temperature-dataset", methods=['POST'])
@cross_origin()
def predict_temperature_dataset():
    try:
        filePath=request.files['filePath']
        fileName='Prediction_Regression.csv'

        if not os.path.isdir('Prediction_Files'):
            os.mkdir('Prediction_Files')
        path='Prediction_Files'+'/'+fileName
        filePath.save(path)

        pred = predictionRegression(path)
        output_folder=pred.predictionFromModelRegression()
        os.remove(path)
        return send_file(output_folder, as_attachment=True)
    except Exception:
        flash('Something went wrong', 'danger')
        return redirect(url_for('temperature'))




if __name__ == "__main__":
    app.run()