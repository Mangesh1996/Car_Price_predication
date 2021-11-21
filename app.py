from flask import Flask, render_template,request
from numpy.core.fromnumeric import sort
import pandas as pd
import pickle
app = Flask(__name__)
car=pd.read_csv("cleancar.csv")
model=pickle.load(open("LinearRegressionmodel.pkl","rb"))


@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['company'].unique())
    companies.insert(0,"Select a Company")
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()
    return render_template('index.html',companies=companies,car_models=car_models,years=year,fuel_type=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    
    car_models=request.form.get('car_models')
    year=int(request.form.get('year'))
    fuel=request.form.get('fuel')
    kilo_driven=int(request.form.get('kilo_driven'))
    print(company,car_models,year,fuel,kilo_driven)
    predication=model.predict(pd.DataFrame([[car_models,company,year,kilo_driven,fuel]],columns=['name','company','year','kms_driven','fuel_type']))

    return str(round(predication[0],2))
if __name__ == "__main__":
    app.run(debug=True)
