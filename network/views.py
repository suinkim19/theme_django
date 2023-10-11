from django.shortcuts import render
from django_plotly_dash import DjangoDash
import importlib

# Create your views here.
#def index(request):
#    return render(request, 'network/index.html')

def load_dash_app(request, week):
    week_module = importlib.import_module(f'network.dash_apps.{week}')
    app_name = f"dash_app_{week}"

    return render(request, 'network/index.html', {'app_name': app_name})

def select_week(request):
    return render(request, 'network/select_week.html')