#https://predictivehacks.com/practical-guide-build-and-deploy-a-machine-learning-web-app/#3
from flask import Flask, render_template, make_response, request, redirect, url_for, send_file
import joblib
import pandas as pd
import numpy as np
import os

from predictions_file import predict_bills

app = Flask(__name__)
app.config["DEBUG"] = True
loaded_model = joblib.load("lrmodel.sav")

@app.route('/')
def index():
    return render_template('home.html')
#This first part give user interface where the home.html file display, the uplaod and submit button.

@app.route('/', methods=['GET', 'POST'])
def uploadFiles():
    uploaded_file = request.files['file']
    #request.files allow the utilisator to import a file and send/post it to the server.
    #Note that 'file' is not the filename; it's the name of the form field (home.html --> name="file") containing the file.
    
    if uploaded_file.filename != '':
    #uploaded_file now contain the file because it uploaded it but if there are no uploaded file,
    #there are no way to know the name of the file for the api.
    #So, xyz.filename is an attribut that know the file name that was uploaded, and the if statement,
    #above say that if the filename of the uploaded file is different than '' so, there are a file uploaded. 
        file_path = ( "file.csv")
        uploaded_file.save(file_path)
    return redirect(url_for('downloadFile'))
    #And we give the filename with it's extensions (.csv) will be stored in a varaible.
    #Then with the .save attribut we save it.
#now we are reading the file, make predictions with our model and save the predictions.
#then we are sending the CSV with the predictions to the user as attachement 
@app.route('/download')
def downloadFile ():
    path = "file.csv"
    #Now we call back the file for use it here in the page where we can give the file inside the model.
    predictions = predict_bills(pd.read_csv(path))
    #We create a variable that use the predict_bills function in order to read the csv file that we upload.
    #For more informations about this function go to predictions_file.py.
    predictions.to_csv('predictions.csv',index=False)
    #Then we save as a csv file the results.
    return send_file("predictions.csv", as_attachment=True)
    #And thanks to the return and send_file function we dowloaded the file with the results.


if __name__ == "__main__":    
    app.run(debug = True)
    