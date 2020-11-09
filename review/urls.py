from django.urls import path

from .views import GetSentiment

urlpatterns = [
    path('classifiers/classify/', GetSentiment.as_view() , name='index'),
]