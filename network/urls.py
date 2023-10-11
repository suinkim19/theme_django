#from django.urls import path # path 함수를 이용하기 위해서 선언해줍니다.
#from . import views     # from 옆에 .(점)은 현재 폴더(app)를 의미합니다. 즉 현재 폴더에 views.py를 가져옵니다
#from .views import DashView


#urlpatterns = [ path('', DashView.as_view(), name = 'network')] 

# network/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.select_week, name='select_week'),
    path('<str:week>/', views.load_dash_app, name='load_dash_app')
]
