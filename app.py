import numpy as np
import pickle
from flask import Flask, request, render_template
import sklearn
app = Flask(__name__)

model = pickle.load(open("loan_prediction_model.pkl", "rb"))
@app.route("/")
def hello():
    return "Hello World! This is my First deployment"

@app.route("/test")
def deployment():
    return "The deployment is done using the Flask and Spyder"


@app.route("/template")
def home():
    return render_template("index.html")

@app.route("/predict", methods =["POST"])
def predict():
    ApplicantIncome = float(request.form['ApplicantIncome'])
    CoapplicantIncome = float(request.form['CoapplicantIncome'])
    LoanAmount = float(request.form['LoanAmount'])
    Gender = request.form['Gender']
    if(Gender == 'Male'):
        Gender_Male = 1
        Gender_Female = 0
    else:
        Gender_Male = 0
        Gender_Female = 1
    Married = request.form['Married']
    if(Married == 'Yes'):
        Married_Yes = 1
        Married_No = 0
    else:
        Married_Yes = 0
        Married_No = 1
    Dependents = request.form['Dependents']
    if(Dependents == 0):
        Dependents_0  = 1              
        Dependents_1  = 0           
        Dependents_2  = 0              
        Dependents_3above = 0
    elif(Dependents == 1):
        Dependents_0  = 0              
        Dependents_1  = 1           
        Dependents_2  = 0              
        Dependents_3above = 0
    elif(Dependents == 2):
        Dependents_0  = 0             
        Dependents_1  = 0           
        Dependents_2  = 1              
        Dependents_3above = 0
    else:
        Dependents_0  = 0              
        Dependents_1  = 0           
        Dependents_2  = 0              
        Dependents_3above = 1
    Education = request.form['Education']
    if(Education == 'Graduate'):
        Education_Graduate = 1
        Education_NotGraduate = 0
    else:
        Education_Graduate = 0
        Education_NotGraduate = 1
    Self_Employed = request.form['Self_Employed']
    if(Self_Employed == 'Yes'):
        Self_Employed_Yes = 1
        Self_Employed_No = 0
    else:
        Self_Employed_Yes = 0
        Self_Employed_No = 1
    Loan_Amount_Term = request.form['Loan_Amount_Term']  
    if(Loan_Amount_Term==12):
        Loan_Amount_Term_12  = 1   
        Loan_Amount_Term_36  = 0
        Loan_Amount_Term_60  = 0  
        Loan_Amount_Term_84  = 0  
        Loan_Amount_Term_120 = 0  
        Loan_Amount_Term_180 = 0  
        Loan_Amount_Term_240 = 0  
        Loan_Amount_Term_300 = 0 
        Loan_Amount_Term_360 = 0  
        Loan_Amount_Term_480 = 0
    elif(Loan_Amount_Term==36):
        Loan_Amount_Term_12  = 0   
        Loan_Amount_Term_36  = 1
        Loan_Amount_Term_60  = 0  
        Loan_Amount_Term_84  = 0  
        Loan_Amount_Term_120 = 0  
        Loan_Amount_Term_180 = 0  
        Loan_Amount_Term_240 = 0  
        Loan_Amount_Term_300 = 0 
        Loan_Amount_Term_360 = 0  
        Loan_Amount_Term_480 = 0 
    elif(Loan_Amount_Term==60):
        Loan_Amount_Term_12  = 0   
        Loan_Amount_Term_36  = 0
        Loan_Amount_Term_60  = 1  
        Loan_Amount_Term_84  = 0  
        Loan_Amount_Term_120 = 0  
        Loan_Amount_Term_180 = 0  
        Loan_Amount_Term_240 = 0  
        Loan_Amount_Term_300 = 0 
        Loan_Amount_Term_360 = 0  
        Loan_Amount_Term_480 = 0 
    elif(Loan_Amount_Term==84):
        Loan_Amount_Term_12  = 0   
        Loan_Amount_Term_36  = 0
        Loan_Amount_Term_60  = 0 
        Loan_Amount_Term_84  = 1  
        Loan_Amount_Term_120 = 0  
        Loan_Amount_Term_180 = 0  
        Loan_Amount_Term_240 = 0  
        Loan_Amount_Term_300 = 0 
        Loan_Amount_Term_360 = 0  
        Loan_Amount_Term_480 = 0   
    elif(Loan_Amount_Term==120):
        Loan_Amount_Term_12  = 0   
        Loan_Amount_Term_36  = 0
        Loan_Amount_Term_60  = 0 
        Loan_Amount_Term_84  = 0  
        Loan_Amount_Term_120 = 1 
        Loan_Amount_Term_180 = 0  
        Loan_Amount_Term_240 = 0  
        Loan_Amount_Term_300 = 0 
        Loan_Amount_Term_360 = 0  
        Loan_Amount_Term_480 = 0 
    elif(Loan_Amount_Term==180):
        Loan_Amount_Term_12  = 0   
        Loan_Amount_Term_36  = 0
        Loan_Amount_Term_60  = 0 
        Loan_Amount_Term_84  = 0  
        Loan_Amount_Term_120 = 0
        Loan_Amount_Term_180 = 1  
        Loan_Amount_Term_240 = 0  
        Loan_Amount_Term_300 = 0 
        Loan_Amount_Term_360 = 0  
        Loan_Amount_Term_480 = 0
    elif(Loan_Amount_Term==240):
        Loan_Amount_Term_12  = 0   
        Loan_Amount_Term_36  = 0
        Loan_Amount_Term_60  = 0 
        Loan_Amount_Term_84  = 0  
        Loan_Amount_Term_120 = 0 
        Loan_Amount_Term_180 = 0  
        Loan_Amount_Term_240 = 1  
        Loan_Amount_Term_300 = 0 
        Loan_Amount_Term_360 = 0  
        Loan_Amount_Term_480 = 0 
    elif(Loan_Amount_Term==300):
        Loan_Amount_Term_12  = 0   
        Loan_Amount_Term_36  = 0
        Loan_Amount_Term_60  = 0 
        Loan_Amount_Term_84  = 0  
        Loan_Amount_Term_120 = 0
        Loan_Amount_Term_180 = 0  
        Loan_Amount_Term_240 = 0  
        Loan_Amount_Term_300 = 1 
        Loan_Amount_Term_360 = 0  
        Loan_Amount_Term_480 = 0
    elif(Loan_Amount_Term==360):
        Loan_Amount_Term_12  = 0   
        Loan_Amount_Term_36  = 0
        Loan_Amount_Term_60  = 0 
        Loan_Amount_Term_84  = 0  
        Loan_Amount_Term_120 = 0
        Loan_Amount_Term_180 = 0  
        Loan_Amount_Term_240 = 0  
        Loan_Amount_Term_300 = 0 
        Loan_Amount_Term_360 = 1 
        Loan_Amount_Term_480 = 0 
    else:
        Loan_Amount_Term_12  = 0   
        Loan_Amount_Term_36  = 0
        Loan_Amount_Term_60  = 0 
        Loan_Amount_Term_84  = 0  
        Loan_Amount_Term_120 = 0 
        Loan_Amount_Term_180 = 0  
        Loan_Amount_Term_240 = 0  
        Loan_Amount_Term_300 = 0 
        Loan_Amount_Term_360 = 0  
        Loan_Amount_Term_480 = 1 
    Credit_History = request.form['Credit_History']
    if(Credit_History == 0):
        Credit_History_0 = 1
        Credit_History_1 = 0
    else:
        Credit_History_0 = 0
        Credit_History_1 = 1 
    Property_Area = request.form['Property_Area']
    if(Property_Area == 'Rural'):
        Property_Area_Rural = 1
        Property_Area_Semiurban = 0
        Property_Area_Urban = 0
    elif(Property_Area == 'Semiurban'):
        Property_Area_Rural = 0
        Property_Area_Semiurban = 1
        Property_Area_Urban = 0 
    else:
        Property_Area_Rural = 0
        Property_Area_Semiurban = 0
        Property_Area_Urban = 1
    prediction = model.predict_proba([[ApplicantIncome,CoapplicantIncome,LoanAmount,Gender_Male,Gender_Female,Married_Yes,Married_No,Dependents_0,Dependents_1,Dependents_2,Dependents_3above,Education_Graduate,Education_NotGraduate,
Self_Employed_Yes,Self_Employed_No,Loan_Amount_Term_12,Loan_Amount_Term_36,Loan_Amount_Term_60,Loan_Amount_Term_84,Loan_Amount_Term_120,
Loan_Amount_Term_180,Loan_Amount_Term_240,Loan_Amount_Term_300,Loan_Amount_Term_360,Loan_Amount_Term_480,Credit_History_0,Credit_History_1,
Property_Area_Rural, Property_Area_Semiurban,Property_Area_Urban]])[:,0][0]
    
        
    
    # Check the output values and retrieve the result with html tag based on the value
    if prediction > 0.5:
        return render_template("index.html",result = "Congratulatons!You are Eligible for Loan")
    else:
        return render_template("index.html",result = "OOPS!!!Sorry, Not Eligible for Loan!")

if __name__ == "__main__":
    app.run(debug = True)



