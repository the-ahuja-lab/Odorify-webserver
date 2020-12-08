from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
	path('', views.home, name= "olfy-home"),
	path('loading', views.loadingpage, name= "olfy-loading"),
	path('models', views.displaymodels, name= "olfy-models"),
	path('about', views.about, name= "olfy-about"),
	path('help', views.help, name= "olfy-help"),
	path('odor', views.odor, name= "olfy-odor"),
	path('odor2', views.odor2,name= "olfy-odor2"),
	path('odor_OR', views.odor_Or, name= "olfy-odor_or"),
	path('OR', views.Or, name= "olfy-or"),
	path('queue', views.queue, name= "olfy-queue"),
	path('results', views.results, name= "olfy-results"),
	path('results/<str:job_name>/<str:model>/<str:count>/<str:flag>', views.result_queue, name= "olfy-result_queue"),
	path('contact', views.contactus, name= "olfy-contact"),
	path('download/<str:job_name>/<str:model>/<str:count>', views.download, name= 'download'),
	path('download/<str:job_name>/<str:model>/<str:count>/<str:flag>', views.download, name= 'download'),
]