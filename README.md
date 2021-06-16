# prediction
prediction of customer engagement in future

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


data = pd.read_csv('bank.csv')
final_col=['marital', 'default', 'housing', 'day', 'campaign', 'pdays', 'previous','poutcome','contact']
data.drop(columns=final_col,axis='columns',inplace=True)
le_job = LabelEncoder()
data['job']= le_job.fit_transform(data['job'])
le_education = LabelEncoder()
data['education']= le_education.fit_transform(data['education'])
le_loan = LabelEncoder()
data['loan']= le_loan.fit_transform(data['loan'])
le_month = LabelEncoder()
data['month']= le_month.fit_transform(data['month'])
le_y = LabelEncoder()
data['y']= le_y.fit_transform(data['y'])



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age1 = int(request.values['age'])
    job1 = request.values['job']
    job1=le_job.transform([job1])
    edu1 = request.values['education']
    edu1=le_education.transform([edu1])
    bal1 =int(request.values['balance'])
    loan1 = request.values['loan']
    loan1=le_loan.transform([loan1])
    month1 = request.values['month']
    month1=le_month.transform([month1])
    dur1 = int(request.values['duration'])

    row=[age1,job1[0],edu1[0],bal1,loan1[0],month1[0],dur1]   
    
    final_features = [np.array(row)]
    prediction = model.predict( final_features)

    if prediction == 1:
        pred = "The person is more likely to apply for the scheme."
    elif prediction == 0:
        pred = "The person won't apply for the scheme."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
