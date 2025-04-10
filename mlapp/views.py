from django.shortcuts import render
import pickle
import pandas as pd

#load model once
#model location and rb = read binary
with open('mlapp/model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_view(request):
    prediction = None

    if request.method == 'POST':
        try:
            #create a data dictionary from post data
            data = {
                'gender': request.POST.get('gender'),
                'age': int(request.POST.get('age')),
                'hypertension': int(request.POST.get('hypertension')),
                'heart_disease': int(request.POST.get('heart_disease')),
                'smoking_history': request.POST.get('smoking_history'),
                'bmi': float(request.POST.get('bmi')),
                'HbA1c_level': float(request.POST.get('HbA1c_level')),
                'blood_glucose_level': float(request.POST.get('blood_glucose_level')),
            }
            features = pd.DataFrame([data])
            prediction_result = model.predict(features)
            prediction = prediction_result[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    #return the prediction result in the predict.html
    return render(request, 'mlapp/predict.html', {'prediction':prediction})
        


# Create your views here.
