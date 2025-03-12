import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from .models import *
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.decorators import login_required
import os

# Create your views here.


@login_required(login_url='login')
def start(request):
    return render(request, 'start.html')


@login_required(login_url='login')
def project(request):
    return render(request, 'project.html')

# Define class labels
class_labels = {
    1: "Bean",
    2: "Bitter_Gourd",
    3: "Bottle_Gourd",
    4: "Brinjal",
    5: "Broccoli",
    6: "Cabbage",
    7: "Capsicum",
    8: "Carrot",
    9: "Cauliflower",
    10: "Cucumber",
    11: "Papaya",
    12: "Potato",
    13: "Pumpkin",
    14: "Radish",
    15: "Tomato"
}

# Load the model
model_path = 'D:\\Karunadu Internship\\ProjectComplete\\full\Vegetable Classifier\\my_model.h5'
model = load_model(model_path)


def classify_image(img_path):
    try:
        image = Image.open(img_path)
        image = image.resize((30, 30))
        image = np.expand_dims(image, axis=0)
        image = np.array(image) / 255.0
        predictions = model.predict(image)
        result_index = np.argmax(predictions)
        return class_labels.get(result_index + 1, "Unknown Vegetable")
    except Exception as e:
        return f"Error in classification: {str(e)}"


@login_required(login_url='login')
def index(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        # Get the full path of the uploaded image
        img_path = fs.path(filename)

        # Classify the image
        result = classify_image(img_path)

        return render(request, 'index.html', {
            'uploaded_file_url': uploaded_file_url,
            'result': result
        })

    return render(request, 'index.html')


def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        user = User.objects.filter(username=username)
        user = authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('start')
        else:
            result = "Password Entered is wrong"
            return HttpResponse("Username or Password is incorrect!!!")

    return render(request, 'login.html')


def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')
        if pass1 != pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:
            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()
            return redirect('login')

    return render(request, 'signup.html')


def LogoutPage(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def intro(request):
    if (request.method == "POST"):
        data = request.POST
        alpha = float(data.get('alpha'))
        delta = float(data.get('delta'))
        u = float(data.get('u'))
        g = float(data.get('g'))
        r = float(data.get('r'))
        i = float(data.get('i'))
        z = float(data.get('z'))
        run_ID = float(data.get('run_ID'))
        rerun_ID = float(data.get('rerun_ID'))
        cam_col = float(data.get('cam_col'))
        field_ID = float(data.get('field_ID'))
        spec_obj_ID = float(data.get('spec_obj_ID'))
        redshift = float(data.get('redshift'))
        plate = float(data.get('plate'))
        MJD = float(data.get('MJD'))
        fiber_ID = float(data.get('fiber_ID'))
        if ('buttonn' in request.POST):
            import pandas as pd
            import matplotlib.pyplot as plt
            import sklearn
            from sklearn.model_selection import train_test_split
            import numpy as np

            path = "D:\\Karunadu Internship\\ProjectComplete\\full\\ML\\train_dataset.csv"
            data = pd.read_csv(path)

            output = data.drop(['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID', 'rerun_ID','cam_col', 'field_ID', 'spec_obj_ID', 'redshift', 'plate', 'MJD', 'fiber_ID'], axis=1)
            inputs = data.drop(['class'], axis=1)

            x_train, x_test, y_train, y_test = train_test_split(
                inputs, output, train_size=0.8)

            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            model.fit(x_train, y_train)

            res = model.predict([[alpha, delta, u, g, r, i, z, run_ID, rerun_ID,cam_col, field_ID, spec_obj_ID, redshift, plate, MJD, fiber_ID,]])
            return render(request, 'h1.html', context={'result': 'Predicted class is ' +res[0]})
    return render(request, 'D:\\Karunadu Internship\\ProjectComplete\\full\\app\\templates\\h1.html')
