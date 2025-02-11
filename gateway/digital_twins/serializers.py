from rest_framework import serializers
from smile_ml_core.classes.scoring import MetricName

from digital_twins.models import DigitalTwin, Model
from polygon.models import Competition
from project.models import Project
from project.serializers import BaseModelSerializer
from project.services.scores_table import get_best_module_id, get_model_node_id


class DigitalTwinSerializer(BaseModelSerializer):
    class Meta:
        model = DigitalTwin
        fields = ['id', 'name', 'description']


class ModelSerializer(BaseModelSerializer):
    class Meta:
        model = Model
        fields = ['id', 'name', 'description']


class DigitalTwinDetailSerializer(BaseModelSerializer):
    models = ModelSerializer(read_only=True, many=True)

    class Meta:
        model = DigitalTwin
        fields = ['id', 'name', 'description', 'models']


class DigitalTwinUpdateSerializer(serializers.Serializer):
    competition_ids = serializers.ListField(child=serializers.IntegerField(), write_only=True)

    def update(self, instance, validated_data):
        user = validated_data['user']

        for competition_id in validated_data['competition_ids']:
            competition = Competition.objects.get(id=competition_id)
            module_baseline_ids = competition.baseline_module.values_list('id', flat=True)
            metric_name = MetricName(competition.metric).name.lower()

            module_id, _ = get_best_module_id(module_baseline_ids, metric_name, user_id=user.id)
            module = Project.objects.get(id=module_id)

            model_name = (
                'FedotModel'
                if module.extra_params_surrogate.include_auto_ml
                else module.extra_params_surrogate.cv_model
            )
            node_id = get_model_node_id(module_id, user.id, model_name)

            model = Model(
                name=competition.name,
                description=competition.project.description,
                user=user,
                digital_twin=instance,
                competition=competition,
                module=module,
                node_id=node_id,
            )
            model.save()

        return instance


class SimpleCompetitionSerializer(BaseModelSerializer):
    class Meta:
        model = Competition
        fields = ['id', 'name']


class CompetitionsResponseSerializer(serializers.Serializer):
    competitions = SimpleCompetitionSerializer(many=True)


class FeaturesResponseSerializer(serializers.Serializer):
    features = serializers.CharField()
    target_column = serializers.CharField()


class ModelOneCalcSerializer(serializers.Serializer):
    params = serializers.JSONField()


class ModelOneCalcResponseSerializer(serializers.Serializer):
    predict = serializers.FloatField()


class ModelDataCalcSerializer(serializers.Serializer):
    file = serializers.FileField()
