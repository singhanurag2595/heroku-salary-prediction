from flask import Flask, render_template, request
import joblib

model = joblib.load('hiring_model.pkl')

app=Flask(__name__)

@app.route('/')
def main_func():
    return render_template('model_base.html',)

@app.route('/predict' , methods = ['POST'])
def predict():

    exp = request.form.get('experience')
    score = request.form.get('test_score')
    interview_score = request.form.get('interview_score')

    prediction = model.predict([[int(exp) , int(score) , int(interview_score)]])

    output = round(prediction[0] , 2)

    return render_template('model_base.html' , prediction_text = f"Employee Salary will be $ {output}")

app.run(debug=True)