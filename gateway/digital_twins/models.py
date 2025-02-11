from django.db import models

from polygon.models import Competition
from project.models import Project, ProjectEntity


class DigitalTwin(ProjectEntity):
    ...


class Model(ProjectEntity):
    digital_twin = models.ForeignKey(DigitalTwin, on_delete=models.CASCADE, related_name='models')
    competition = models.ForeignKey(Competition, on_delete=models.CASCADE)
    module = models.ForeignKey(Project, on_delete=models.CASCADE)
    node_id = models.CharField()
