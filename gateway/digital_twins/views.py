import pickle

import pandas as pd
from django.core.files.base import ContentFile
from django.db.models import Q
from django.http import JsonResponse
from drf_spectacular.utils import extend_schema
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from smile_ml_core.constants import ROUNDING
from smile_ml_core.data.tools import DataTuplesList

from digital_twins.models import DigitalTwin, Model
from digital_twins.serializers import (
    CompetitionsResponseSerializer,
    DigitalTwinDetailSerializer,
    DigitalTwinSerializer,
    DigitalTwinUpdateSerializer,
    FeaturesResponseSerializer,
    ModelDataCalcSerializer,
    ModelOneCalcResponseSerializer,
    ModelOneCalcSerializer,
    ModelSerializer,
    SimpleCompetitionSerializer,
)
from file.models import File
from file.reader import handle_file
from file.tasks import generate_file_link
from polygon.models import Competition, TaskEnum
from services.clients.graph.v1 import graph_client
from services.schemas.graph import SavedFilePath
from userprofile.models import UserProfile


class APIDigitalTwinModelViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user.userprofile
        queryset = DigitalTwin.objects.filter(user=user)
        return queryset.order_by('-pk')

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return DigitalTwinDetailSerializer
        if self.action == 'update':
            return DigitalTwinUpdateSerializer
        if self.action == 'get_competitions':
            return None

        return DigitalTwinSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user.userprofile)

    def perform_update(self, serializer):
        serializer.save(user=self.request.user.userprofile)

    @extend_schema(responses=CompetitionsResponseSerializer)
    @action(methods=['GET'], url_path='competitions', detail=False)
    def get_competitions(self, request: Request):
        user = self.request.user.userprofile
        queryset = Competition.objects.filter(Q(owner=user) & Q(task=TaskEnum.SURROGATE_MODELING))
        serializer = SimpleCompetitionSerializer(queryset, many=True)
        return Response({'competitions': serializer.data})


class APIDigitalTwinModelModelViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user.userprofile
        queryset = Model.objects.filter(user=user)
        return queryset.order_by('-pk')

    def get_serializer_class(self):
        if self.action == 'one_calculation':
            return ModelOneCalcSerializer
        if self.action == 'data_calculation':
            return ModelDataCalcSerializer

        return ModelSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user.userprofile)

    def perform_update(self, serializer):
        serializer.save(user=self.request.user.userprofile)

    @extend_schema(responses=FeaturesResponseSerializer)
    @action(methods=['GET'], url_path='features', detail=True)
    def get_features(self, request: Request, pk: int):
        model = get_object_or_404(Model, pk=pk)
        data = {
            'features': model.competition.features,
            'target_column': model.competition.target_column,
        }
        return Response(data, status=status.HTTP_200_OK)

    @extend_schema(responses=ModelOneCalcResponseSerializer)
    @action(methods=['POST'], url_path='one_calculation', detail=True)
    def one_calculation(self, request: Request, pk: int):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        params = serializer.validated_data['params']
        df = pd.DataFrame.from_dict(params, orient='index', dtype=float).T

        result = apply_to_data(model_id=pk, df=df, filename='tmp_file', user_id=request.user.pk)
        predict = result['predict']['0']

        return Response({'predict': predict}, status=status.HTTP_200_OK)

    @action(methods=['POST'], url_path='data_calculation', detail=True)
    def data_calculation(self, request: Request, pk) -> JsonResponse:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        files: DataTuplesList = handle_file(file=serializer.validated_data['file'])

        file = files[0]
        name, data = file

        model = get_object_or_404(Model, pk=pk)
        predict = apply_to_data(model_id=model.pk, df=data, user_id=request.user.pk, filename=name)

        df_pred = pd.DataFrame.from_dict(predict)
        df_pred.index = pd.to_numeric(df_pred.index)
        df_pred = df_pred.rename(columns={'predict': model.competition.target_column})
        df_pred = df_pred.round(ROUNDING)

        df = data.join(df_pred)

        userprofile = get_object_or_404(UserProfile, user=request.user)

        file_obj = File.objects.create(
            project_owner=model.module,
            user_uploaded=userprofile,
            file=ContentFile(pickle.dumps(df, protocol=4), name),
        )

        generate_file_link.delay(file_id=file_obj.pk, filename=file_obj.file.name, user_id=request.user.pk)

        df_dict = df.to_dict(orient='records')
        data = {'file_id': file_obj.pk, 'df_dict': df_dict}

        return JsonResponse(data, safe=False, status=status.HTTP_200_OK)


def apply_to_data(model_id: int, df: pd.DataFrame, user_id: int, filename: str) -> dict[str, dict[str, float]]:
    userprofile = get_object_or_404(UserProfile, user_id=user_id)
    model = get_object_or_404(Model, pk=model_id)

    file_obj = File.objects.create(
        project_owner=model.module,
        user_uploaded=userprofile,
        file=ContentFile(pickle.dumps(df, protocol=4), filename),
    )
    file_path = SavedFilePath(file=file_obj.file.url, user_id=user_id, module_id=model.module.pk)
    result = graph_client.model_nodes.apply_to_new_data(node_id=model.node_id, payload=file_path)

    file_obj.delete()

    return result
