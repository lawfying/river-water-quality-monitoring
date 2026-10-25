from rest_framework import serializers
from .models import Sensor, SensorReading, AnalysisReport, WaterRecord


class SensorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sensor
        fields = ['id', 'name', 'device_id', 'location', 'location_detail',
                  'status', 'last_active', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class SensorReadingSerializer(serializers.ModelSerializer):
    """写入用: 通过 device_id 关联传感器"""
    device_id = serializers.CharField(write_only=True)

    class Meta:
        model = SensorReading
        fields = ['id', 'device_id', 'ph', 'hardness', 'solids', 'chloramines',
                  'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes',
                  'turbidity', 'temperature', 'flow_rate', 'water_level',
                  'recorded_at', 'created_at']
        read_only_fields = ['id', 'created_at']

    def validate_device_id(self, value):
        if not Sensor.objects.filter(device_id=value).exists():
            raise serializers.ValidationError(f'传感器 {value} 不存在')
        return value

    def create(self, validated_data):
        device_id = validated_data.pop('device_id')
        sensor = Sensor.objects.get(device_id=device_id)
        validated_data['sensor'] = sensor
        # 更新传感器状态
        from django.utils import timezone
        Sensor.objects.filter(id=sensor.id).update(
            status='online', last_active=timezone.now()
        )
        return super().create(validated_data)


class SensorReadingReadSerializer(serializers.ModelSerializer):
    """读取用: 嵌套传感器名称"""
    sensor_name = serializers.CharField(source='sensor.name', read_only=True)
    sensor_device_id = serializers.CharField(source='sensor.device_id', read_only=True)
    sensor_location = serializers.CharField(source='sensor.get_location_display', read_only=True)

    class Meta:
        model = SensorReading
        fields = ['id', 'sensor_name', 'sensor_device_id', 'sensor_location',
                  'ph', 'hardness', 'solids', 'chloramines',
                  'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes',
                  'turbidity', 'temperature', 'flow_rate', 'water_level',
                  'recorded_at', 'created_at']


class SensorReportSerializer(serializers.Serializer):
    """传感器报告查询参数"""
    device_id = serializers.CharField(required=False)
    hours = serializers.IntegerField(default=24)


class AnalysisReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisReport
        fields = ['id', 'title', 'content', 'summary', 'wqi_score', 'wqi_grade',
                  'water_data', 'source', 'created_at']
        read_only_fields = ['id', 'created_at']


class WaterRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = WaterRecord
        fields = ['sequence_id', 'ph', 'hardness', 'solids', 'chloramines',
                  'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes',
                  'turbidity', 'wqi_score', 'wqi_label', 'rf_prediction',
                  'source', 'created_at']
        read_only_fields = ['sequence_id', 'created_at']
