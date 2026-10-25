from django.utils import timezone
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Avg
from datetime import timedelta

from .models import Sensor, SensorReading, AnalysisReport, WaterRecord
from .serializers import (
    SensorSerializer,
    SensorReadingSerializer,
    SensorReadingReadSerializer,
    SensorReportSerializer,
    AnalysisReportSerializer,
    WaterRecordSerializer,
)


class SensorViewSet(viewsets.ModelViewSet):
    """传感器 CRUD"""
    queryset = Sensor.objects.all()
    serializer_class = SensorSerializer
    lookup_field = 'device_id'
    lookup_url_kwarg = 'device_id'

    def get_queryset(self):
        qs = super().get_queryset()
        location = self.request.query_params.get('location')
        status_param = self.request.query_params.get('status')
        if location:
            qs = qs.filter(location=location)
        if status_param:
            qs = qs.filter(status=status_param)
        return qs


class WaterRecordViewSet(viewsets.ModelViewSet):
    """水质监测记录 CRUD"""
    queryset = WaterRecord.objects.all()
    serializer_class = WaterRecordSerializer
    ordering = ['-sequence_id']

    def get_queryset(self):
        qs = super().get_queryset()
        limit = self.request.query_params.get('limit')
        if limit:
            try:
                qs = qs[:int(limit)]
            except ValueError:
                pass
        return qs

    @action(detail=False, methods=['post'], url_path='batch-delete')
    def batch_delete(self, request):
        """批量删除报告"""
        ids = request.data.get('ids', [])
        if not ids:
            return Response({'error': '请提供要删除的ID列表'}, status=status.HTTP_400_BAD_REQUEST)
        deleted, _ = AnalysisReport.objects.filter(id__in=ids).delete()
        return Response({'deleted_count': deleted})

    @action(detail=False, methods=['get'], url_path='stats')
    def stats(self, request):
        """传感器统计概览"""
        total = Sensor.objects.count()
        online = Sensor.objects.filter(status='online').count()
        offline = Sensor.objects.filter(status='offline').count()
        error = Sensor.objects.filter(status='error').count()
        return Response({
            'total': total,
            'online': online,
            'offline': offline,
            'error': error,
        })


class SensorReadingViewSet(viewsets.ModelViewSet):
    """传感器读数接口"""
    queryset = SensorReading.objects.select_related('sensor').all()
    ordering = ['-recorded_at']

    def get_serializer_class(self):
        if self.action == 'list' or self.action == 'retrieve':
            return SensorReadingReadSerializer
        return SensorReadingSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        device_id = self.request.query_params.get('device_id')
        hours = self.request.query_params.get('hours')
        if device_id:
            qs = qs.filter(sensor__device_id=device_id)
        if hours:
            try:
                hours = int(hours)
                since = timezone.now() - timedelta(hours=hours)
                qs = qs.filter(recorded_at__gte=since)
            except ValueError:
                pass
        return qs

    @action(detail=False, methods=['get'], url_path='latest')
    def latest(self, request):
        """返回每个传感器的最新一条数据"""
        device_id = request.query_params.get('device_id')
        sensors = Sensor.objects.all()
        if device_id:
            sensors = sensors.filter(device_id=device_id)

        results = []
        for sensor in sensors:
            reading = SensorReading.objects.filter(sensor=sensor).order_by('-recorded_at').first()
            if reading:
                ser = SensorReadingReadSerializer(reading)
                results.append(ser.data)

        return Response(results)

    @action(detail=False, methods=['get'], url_path='report')
    def report(self, request):
        """返回时间段内各指标平均值"""
        ser = SensorReportSerializer(data=request.query_params)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        qs = SensorReading.objects.all()
        if data.get('device_id'):
            qs = qs.filter(sensor__device_id=data['device_id'])
        since = timezone.now() - timedelta(hours=data['hours'])
        qs = qs.filter(recorded_at__gte=since)

        agg = qs.aggregate(
            avg_ph=Avg('ph'),
            avg_hardness=Avg('hardness'),
            avg_solids=Avg('solids'),
            avg_chloramines=Avg('chloramines'),
            avg_sulfate=Avg('sulfate'),
            avg_conductivity=Avg('conductivity'),
            avg_organic_carbon=Avg('organic_carbon'),
            avg_trihalomethanes=Avg('trihalomethanes'),
            avg_turbidity=Avg('turbidity'),
            avg_temperature=Avg('temperature'),
            avg_flow_rate=Avg('flow_rate'),
            avg_water_level=Avg('water_level'),
        )
        agg['count'] = qs.count()
        agg['hours'] = data['hours']
        if data.get('device_id'):
            agg['device_id'] = data['device_id']

        return Response(agg)

    @action(detail=False, methods=['post'], url_path='batch')
    def batch_create(self, request):
        """批量上报传感器数据"""
        if not isinstance(request.data, list):
            return Response({'error': '需要提供数组'}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        errors = []
        for idx, item in enumerate(request.data):
            ser = SensorReadingSerializer(data=item)
            if ser.is_valid():
                ser.save()
                results.append(ser.data)
            else:
                errors.append({'index': idx, 'errors': ser.errors})

        return Response({
            'success_count': len(results),
            'error_count': len(errors),
            'errors': errors,
        })


class AnalysisReportViewSet(viewsets.ModelViewSet):
    """LLM 分析报告 CRUD"""
    queryset = AnalysisReport.objects.all()
    serializer_class = AnalysisReportSerializer
    ordering = ['-created_at']

    def get_queryset(self):
        qs = super().get_queryset()
        limit = self.request.query_params.get('limit')
        if limit:
            try:
                qs = qs[:int(limit)]
            except ValueError:
                pass
        return qs
