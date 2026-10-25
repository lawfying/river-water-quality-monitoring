from django.db import models


class Sensor(models.Model):
    """传感器设备"""
    STATUS_CHOICES = [
        ('online', '在线'),
        ('offline', '离线'),
        ('error', '故障'),
    ]
    LOCATION_CHOICES = [
        ('inlet', '取水口'),
        ('outlet', '出水口'),
        ('midstream', '河道中游'),
        ('reservoir', '水库'),
        ('other', '其他'),
    ]

    name = models.CharField(max_length=100, verbose_name='传感器名称')
    device_id = models.CharField(max_length=50, unique=True, verbose_name='设备编号')
    location = models.CharField(max_length=20, choices=LOCATION_CHOICES, default='other', verbose_name='安装位置')
    location_detail = models.CharField(max_length=200, blank=True, verbose_name='位置详情')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='offline', verbose_name='状态')
    last_active = models.DateTimeField(null=True, blank=True, verbose_name='最后通信时间')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '传感器'
        verbose_name_plural = '传感器'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.name} ({self.device_id})'


class SensorReading(models.Model):
    """传感器读数"""
    sensor = models.ForeignKey(Sensor, on_delete=models.CASCADE, related_name='readings', verbose_name='传感器')
    # 水质指标
    ph = models.FloatField(null=True, blank=True, verbose_name='pH值')
    hardness = models.FloatField(null=True, blank=True, verbose_name='硬度 (mg/L)')
    solids = models.FloatField(null=True, blank=True, verbose_name='总溶解固体 (ppm)')
    chloramines = models.FloatField(null=True, blank=True, verbose_name='氯胺 (mg/L)')
    sulfate = models.FloatField(null=True, blank=True, verbose_name='硫酸盐 (mg/L)')
    conductivity = models.FloatField(null=True, blank=True, verbose_name='电导率 (μS/cm)')
    organic_carbon = models.FloatField(null=True, blank=True, verbose_name='有机碳 (mg/L)')
    trihalomethanes = models.FloatField(null=True, blank=True, verbose_name='三卤甲烷 (μg/L)')
    turbidity = models.FloatField(null=True, blank=True, verbose_name='浊度 (NTU)')
    # 额外环境参数
    temperature = models.FloatField(null=True, blank=True, verbose_name='温度 (°C)')
    flow_rate = models.FloatField(null=True, blank=True, verbose_name='流速 (m/s)')
    water_level = models.FloatField(null=True, blank=True, verbose_name='水位 (m)')
    # 元数据
    recorded_at = models.DateTimeField(verbose_name='采集时间')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='记录创建时间')

    class Meta:
        verbose_name = '传感器读数'
        verbose_name_plural = '传感器读数'
        ordering = ['-recorded_at']
        indexes = [
            models.Index(fields=['sensor', '-recorded_at']),
        ]

    def __str__(self):
        return f'{self.sensor.name} @ {self.recorded_at}'


class AnalysisReport(models.Model):
    """LLM 分析报告"""
    title = models.CharField(max_length=200, verbose_name='报告标题')
    content = models.TextField(verbose_name='报告内容')
    summary = models.CharField(max_length=500, blank=True, verbose_name='摘要')
    wqi_score = models.FloatField(null=True, blank=True, verbose_name='WQI评分')
    wqi_grade = models.CharField(max_length=10, blank=True, verbose_name='WQI等级')
    water_data = models.JSONField(null=True, blank=True, verbose_name='水质数据快照')
    source = models.CharField(max_length=50, default='manual', verbose_name='来源')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '分析报告'
        verbose_name_plural = '分析报告'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.title} ({self.created_at.strftime("%Y-%m-%d %H:%M")})'


class WaterRecord(models.Model):
    """水质监测记录（独立存储，带自增序号）"""
    sequence_id = models.AutoField(primary_key=True, verbose_name='序号')
    # 9项水质指标
    ph = models.FloatField(verbose_name='pH值')
    hardness = models.FloatField(verbose_name='硬度 (mg/L)')
    solids = models.FloatField(verbose_name='总溶解固体 (ppm)')
    chloramines = models.FloatField(verbose_name='氯胺 (mg/L)')
    sulfate = models.FloatField(verbose_name='硫酸盐 (mg/L)')
    conductivity = models.FloatField(verbose_name='电导率 (μS/cm)')
    organic_carbon = models.FloatField(verbose_name='有机碳 (mg/L)')
    trihalomethanes = models.FloatField(verbose_name='三卤甲烷 (μg/L)')
    turbidity = models.FloatField(verbose_name='浊度 (NTU)')
    # 评分
    wqi_score = models.FloatField(null=True, blank=True, verbose_name='WQI综合评分')
    wqi_label = models.CharField(max_length=20, blank=True, verbose_name='水质等级')
    rf_prediction = models.CharField(max_length=10, blank=True, verbose_name='RF预测结果')
    # 元数据
    source = models.CharField(max_length=50, default='manual', verbose_name='数据来源')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='记录时间')

    class Meta:
        verbose_name = '水质记录'
        verbose_name_plural = '水质记录'
        ordering = ['-sequence_id']

    def __str__(self):
        return f'水质记录 #{self.sequence_id} @ {self.created_at.strftime("%Y-%m-%d %H:%M")}'
