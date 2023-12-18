from flask import Flask, jsonify, request, render_template
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64
from io import BytesIO
from shortcut import predict_pca_newdata

app = Flask(__name__)

@app.route('/homepage')
def index():
    return render_template('home.html')


@app.route('/testing', methods=['GET', 'POST'])
def upload_form():
    if request.method == 'POST':
        Faculty = request.form['Faculty']
        Gender = request.form['Gender']
        Year = request.form['Year']
        Q1 = int(request.form['Q1'])
        Q2 = int(request.form['Q2'])
        Q3 = int(request.form['Q3'])
        Q4 = int(request.form['Q4'])
        Q5 = int(request.form['Q5'])
        Q6 = int(request.form['Q6'])
        Q7 = int(request.form['Q7'])
        Q8 = int(request.form['Q8'])
        Q9 = int(request.form['Q9'])
        Q10 = int(request.form['Q10'])
        Q11 = int(request.form['Q11'])
        Q12 = int(request.form['Q12'])
        Q13 = int(request.form['Q13'])
        Q14 = int(request.form['Q14'])
        Q15 = int(request.form['Q15'])
        Q16 = int(request.form['Q16'])
        Q17 = int(request.form['Q17'])
        Q18 = int(request.form['Q18'])
        Q19 = int(request.form['Q19'])
        Q20 = int(request.form['Q20'])
        Q21 = int(request.form['Q21'])
        Q22 = int(request.form['Q22'])
        Q23 = int(request.form['Q23'])
        Q24 = int(request.form['Q24'])
        Q25 = int(request.form['Q25'])
        Q26 = int(request.form['Q26'])
        Q27 = int(request.form['Q27'])
        Q28 = int(request.form['Q28'])
        Q29 = int(request.form['Q29'])
        Q30 = int(request.form['Q30'])
        Q31 = int(request.form['Q31'])
        Q32 = int(request.form['Q32'])
        Q33 = int(request.form['Q33'])
        Q34 = int(request.form['Q34'])
        Q35 = int(request.form['Q35'])

        df = pd.DataFrame({
        'Faculty': [Faculty],
        'Gender': [Gender],
        'Year': [Year],
        'Q1': [Q1],'Q2': [Q2],'Q3': [Q3],'Q4': [Q4],'Q5': [Q5],
        'Q6': [Q6],'Q7': [Q7],'Q8': [Q8],'Q9': [Q9],'Q10': [Q10],
        'Q11': [Q11],'Q12': [Q12],'Q13': [Q13],'Q14': [Q14],'Q15': [Q15],
        'Q16': [Q16],'Q17': [Q17],'Q18': [Q18],'Q19': [Q19],'Q20': [Q20],
        'Q21': [Q21],'Q22': [Q22],'Q23': [Q23],'Q24': [Q24],'Q25': [Q25],
        'Q26': [Q26],'Q27': [Q27],'Q28': [Q28],'Q29': [Q29],'Q30': [Q30],
        'Q31': [Q31],'Q32': [Q32],'Q33': [Q33],'Q34': [Q34],'Q35': [Q35],})
        return 'Files uploaded successfully!'
    else:
        return render_template('testing.html')





@app.route('/result_testing', methods=['POST'])
def alonecluster_data(): 
    Faculty = request.form['Faculty']
    Gender = request.form['Gender']
    Year = request.form['Year']
    Q1 = int(request.form['Q1'])
    Q2 = int(request.form['Q2'])
    Q3 = int(request.form['Q3'])
    Q4 = int(request.form['Q4'])
    Q5 = int(request.form['Q5'])
    Q6 = int(request.form['Q6'])
    Q7 = int(request.form['Q7'])
    Q8 = int(request.form['Q8'])
    Q9 = int(request.form['Q9'])
    Q10 = int(request.form['Q10'])
    Q11 = int(request.form['Q11'])
    Q12 = int(request.form['Q12'])
    Q13 = int(request.form['Q13'])
    Q14 = int(request.form['Q14'])
    Q15 = int(request.form['Q15'])
    Q16 = int(request.form['Q16'])
    Q17 = int(request.form['Q17'])
    Q18 = int(request.form['Q18'])
    Q19 = int(request.form['Q19'])
    Q20 = int(request.form['Q20'])
    Q21 = int(request.form['Q21'])
    Q22 = int(request.form['Q22'])
    Q23 = int(request.form['Q23'])
    Q24 = int(request.form['Q24'])
    Q25 = int(request.form['Q25'])
    Q26 = int(request.form['Q26'])
    Q27 = int(request.form['Q27'])
    Q28 = int(request.form['Q28'])
    Q29 = int(request.form['Q29'])
    Q30 = int(request.form['Q30'])
    Q31 = int(request.form['Q31'])
    Q32 = int(request.form['Q32'])
    Q33 = int(request.form['Q33'])
    Q34 = int(request.form['Q34'])
    Q35 = int(request.form['Q35'])

    df = pd.DataFrame({
        'Faculty': [Faculty],
        'Gender': [Gender],
        'Year': [Year],
        'Q1': [Q1],'Q2': [Q2],'Q3': [Q3],'Q4': [Q4],'Q5': [Q5],
        'Q6': [Q6],'Q7': [Q7],'Q8': [Q8],'Q9': [Q9],'Q10': [Q10],
        'Q11': [Q11],'Q12': [Q12],'Q13': [Q13],'Q14': [Q14],'Q15': [Q15],
        'Q16': [Q16],'Q17': [Q17],'Q18': [Q18],'Q19': [Q19],'Q20': [Q20],
        'Q21': [Q21],'Q22': [Q22],'Q23': [Q23],'Q24': [Q24],'Q25': [Q25],
        'Q26': [Q26],'Q27': [Q27],'Q28': [Q28],'Q29': [Q29],'Q30': [Q30],
        'Q31': [Q31],'Q32': [Q32],'Q33': [Q33],'Q34': [Q34],'Q35': [Q35]})

    file_path = '/Users/phattharakit_/Desktop/application/Stress_Clustering.pkl'
    df = predict_pca_newdata(df, file_path)
    # try:
    #     with open(file_path, 'rb') as file:
    #         existing_data = pickle.load(file)
    # except FileNotFoundError:
    #     existing_data = pd.DataFrame()
    # updated_data = pd.concat([existing_data, df], ignore_index=True) # Add new data to DataFrame 
    # with open(file_path, 'wb') as file: # Save DataFrame into Pickle
    #     pickle.dump(updated_data, file)
    return render_template('result_template.html', data=df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)