from rest_framework.routers import DefaultRouter

from digital_twins import views

router = DefaultRouter()
router.register('digital_twins', views.APIDigitalTwinModelViewSet, basename='digital_twins')
router.register('digital_twin_model', views.APIDigitalTwinModelModelViewSet, basename='digital_twin_model')

urlpatterns = router.urls
