from django.urls import path
from aiapp1 import views

urlpatterns = [
    path('', views.home, name='home'),
]
