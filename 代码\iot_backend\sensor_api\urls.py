from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'sensors', views.SensorViewSet)
router.register(r'readings', views.SensorReadingViewSet)
router.register(r'reports', views.AnalysisReportViewSet)
router.register(r'water-records', views.WaterRecordViewSet)

app_name = 'sensor_api'

urlpatterns = [
    path('', include(router.urls)),
]
