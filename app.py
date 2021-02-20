#import the libraries
import pickle 
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import tensorflow
from PIL import Image  

#Welcome the patients
while(1):
        print("--"*60,sep='\n')
        print(("HEDA - Your Health App").center(120),sep='\n')
        print("--"*60,sep='\n')
        print(("Diagnostics at your door step").center(120),sep='\n')
        print("--"*60,sep='\n')
        print("\n")
        print("List of predictors",sep='\n')
        print("1. Diabetes",sep='\n')
        print("2. Malaria",sep='\n')
        print("3. Heart Disease",sep='\n')
        print("\n")
        predictor= int(input("Please choose your desired predictor number: "))
        print("\n")

        if type(predictor)!=int or 1<predictor>3:
            print("Please enter a number between 1-3")
            continue



        if predictor==1:

            print("**"*60)
            print(("Diabetes prediction").center(120))
            print("**"*60)
            print("\n")
            print("Please Enter the following information: ")

            pregnancies= int(input("Enter number of pregnancies: "))
            glucose= int(input("Enter your glucose level (mg/dl): "))
            blood_pressure= int(input("Enter your blood pressure (mmHg): "))
            skin_thickness= int(input("Enter your skin thickness (mm): "))
            Insulin= int(input("Enter your insulin level (IU/ml): "))
            BMI= int(input("Enter your Body Mass Index (kg/m): "))
            Diabetes_pedigree_function= float(input("Enter number of diabetes pedigree function: "))
            Age= int(input("Enter number of your age (years): "))

            input_list=np.array([[pregnancies, glucose, blood_pressure, skin_thickness, Insulin, BMI, Diabetes_pedigree_function, Age]])
            classifier= pickle.load(open("Diabetes\model.pkl", 'rb'))
            scalar= pickle.load(open("Diabetes\scaler.pkl", 'rb'))
            prediction= classifier.predict(scalar.transform(input_list))

            if prediction==1:
                print("--"*60,sep='\n')
                print(("You have diabetes, Please take care!!!").center(120))
                print("--"*60,sep='\n')
            else:
                print("--"*60,sep='\n')
                print(("Hurray, You don't have diabetes!!!").center(120))
                print("--"*60,sep='\n')
            continue

        if predictor==2:
            print("**"*60)
            print(("Malaria Predictions").center(120))
            print("**"*60)
            
            model=tensorflow.keras.models.load_model("malaria\malaria_model.h5")
            print("The model is loaded")
            print("\n")
            print("Please enter a valid path for the cell image: ")
            print("\n")
            image_path=input("valid path: ")

            print("loading the image")             
            print("\n")                                                                
            img = Image.open(image_path)
            img.show() 
            
            print("predicting the image")
            print("\n")
            img=image.load_img(image_path,target_size=(64,64))
            x=image.img_to_array(img)
            x=x/255
            image_1 = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            prediction=model.predict(image_1)
            print("prediction is completed")
            print("\n")

            if prediction>0.5:
                print("--"*60,sep='\n')
                print(("Hurray, This cell is uninfected!!!").center(120))
                print("--"*60,sep='\n')
            else:
                print("--"*60,sep='\n')
                print(("This cell is parasitized!!!").center(120))
                print("--"*60,sep='\n')
            continue

        if predictor==3:

            print("**"*60)
            print(("Heart Disease prediction").center(100))
            print("**"*60)
            print("\n")
            print("Please Enter the following information: ")

            age= int(input("Enter your age: "))
            sex= int(input("Enter your gender (0 for female/ 1 for male): "))
            chest= int(input("Enter your chest pain type (0: asymptomatic, 1: atypical angina, 2: non-anginal pain, 3: typical angina): "))
            rbp= int(input("Enter your resting blood pressure(mm Hg): "))
            cholestrol= int(input("Enter your Cholestrol (mg/dl): "))
            BSL= int(input("Is your Fasting blood sugar level>120 mg/dl? (0 for no/ 1 for yes): "))
            recg= int(input("Enter your resting ECG value (0,1,2): "))
            heartrate= int(input("Enter the maximum heart rate achieved:"))
            angina= int(input("Do you have exercise induced angina? (0 for no/1 for yes): "))
            oldpeak= float(input("Enter the ST depression induced by exercise relative to rest:"))
            slope= int(input("Enter the slope of the peak exercise ST segment(0: downsloping,1: flat,2: upsloping):"))   
            vessels= int(input("Enter the number of major vessels (0-3) colored by flourosopy:"))
            thal= int(input("Enter type of thalassemia(0 = normal; 1 = fixed defect; 2 = reversable defect):"))

            input_list=np.array([[age,sex,chest,rbp,cholestrol,BSL,recg,heartrate,angina,oldpeak,slope,vessels,thal]])
            classifier= pickle.load(open("Heart Disease Prediction\heartmodel.pkl", 'rb'))
            scalar= pickle.load(open("Heart Disease Prediction\heartscaler.pkl", 'rb'))
            prediction= classifier.predict(scalar.transform(input_list))

            if prediction==1:
                print("--"*60,sep='\n')
                print(("You are at risk for a heart disease. Please obtain medical guidance").center(120))
                print("--"*60,sep='\n')
            else:
                print("--"*60,sep='\n')
                print(("Hurray, Your heart is safe!!!").center(120))
                print("--"*60,sep='\n')
            continue
    