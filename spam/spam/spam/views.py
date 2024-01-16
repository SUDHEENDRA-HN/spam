from django.shortcuts import render
from . import ml_pred

def home(request):
    print("Home view called")
    return render(request, 'index.html')

def res(request):
    text = request.GET.get('text', '')  # Get the 'text' parameter, default to an empty string
    predictions = ml_pred.prediction(text)
    return render(request, 'result.html', {'predictions': predictions})
