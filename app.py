
import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():


    int_features = [x for x in request.form.values()] #Convert string inputs to float.
    
    satisfaction_level = float(int_features[0])
    last_evaluation = float(int_features[1])
    number_project = float(int_features[2])
    average_montly_hours = float(int_features[3])
    time_spend_company = float(int_features[4])
    work_accident = float(int_features[5])
    promotion_last_5years = float(int_features[6])
    first_arr = [satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, work_accident, promotion_last_5years]

    #one hot encoding
    department = int_features[7]
    if(department=='RandD'):
        second_arr = [1,0,0,0,0,0,0,0,0]
    elif(department=='accounting'):
        second_arr = [0,1,0,0,0,0,0,0,0]
    elif(department=='hr'):
        second_arr = [0,0,1,0,0,0,0,0,0]
    elif(department=='management'):
        second_arr = [0,0,0,1,0,0,0,0,0]
    elif(department=='marketing'):
        second_arr = [0,0,0,0,1,0,0,0,0]
    elif(department=='product_mng'):
        second_arr = [0,0,0,0,0,1,0,0,0]
    elif(department=='sales'):
        second_arr = [0,0,0,0,0,0,1,0,0]
    elif(department=='support'):
        second_arr = [0,0,0,0,0,0,0,1,0]
    elif(department=='technical'):
        second_arr = [0,0,0,0,0,0,0,0,1]
    else:
        second_arr = [0,0,0,0,0,0,0,0,0]

    #one hot encoding
    salary = int_features[8]
    if(salary=='low'):
        third_arr = [1,0]
    if(salary=='medium'):
        third_arr = [0,1]
    else:
        third_arr = [0,0]

    features = first_arr + second_arr + third_arr
    prediction =model.predict([features])  # features Must be in the form [[a, b]]
    output = prediction[0]

    return render_template('index.html', prediction_text='Prediction:  {}'.format(output))


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__).  
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()