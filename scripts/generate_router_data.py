"""
生成 BERT Router 训练数据

3 分类：
- gis_term:     GIS 概念/术语解释 → 走 LoRA 模型
- gis_operation: GIS 操作/如何做 → 走 RAG 检索
- general:      闲聊/其他 → 走 LLM 原生
"""

import csv
import os
import random

random.seed(42)

# ===== GIS 术语类 (gis_term) — LoRA 直接回答 =====
gis_term = [
    # 遥感
    "什么是遥感",
    "遥感的原理是什么",
    "什么是卫星遥感",
    "遥感有哪些应用",
    "什么是多光谱遥感",
    "什么是高光谱遥感",
    "雷达遥感是什么",
    "光学遥感和雷达遥感的区别",
    "什么是遥感影像解译",
    "NDVI是什么意思",
    "NDVI怎么计算",
    "遥感中的波段是什么意思",
    "什么是空间分辨率",
    "什么是光谱分辨率",
    "什么是时间分辨率",
    "什么是辐射分辨率",
    "Landsat卫星是什么",
    "MODIS数据是什么",
    "Sentinel卫星系列介绍",
    "SAR是什么",
    "InSAR是什么",
    "LiDAR是什么",
    "什么是点云数据",
    "遥感图像分类有哪些方法",
    "监督分类和非监督分类的区别",
    "什么是遥感影像融合",
    "什么是大气校正",
    "什么是几何校正",
    "什么是正射校正",
    "遥感在农业中的应用",
    "遥感在环境监测中的应用",
    "遥感在灾害评估中的应用",

    # GIS 基础
    "什么是GIS",
    "GIS的全称是什么",
    "GIS有哪些组成部分",
    "GIS的主要功能有哪些",
    "什么是空间分析",
    "什么是缓冲区分析",
    "什么是叠加分析",
    "什么是网络分析",
    "什么是地形分析",
    "地理信息系统和全球定位系统的区别",
    "GIS的发展历史",
    "什么是WebGIS",
    "什么是开源GIS",
    "QGIS和ArcGIS有什么区别",
    "什么是空间数据库",
    "PostGIS是什么",
    "什么是地理编码",
    "什么是逆地理编码",

    # 坐标系统
    "什么是坐标系统",
    "地理坐标系和投影坐标系的区别",
    "WGS84是什么",
    "CGCS2000是什么",
    "北京54坐标系是什么",
    "西安80坐标系是什么",
    "什么是UTM投影",
    "什么是高斯克吕格投影",
    "什么是墨卡托投影",
    "什么是经纬度",
    "什么是大地水准面",
    "什么是椭球体",
    "什么是基准面",
    "什么是地图投影",
    "投影变形的类型有哪些",
    "如何选择合适的投影坐标系",
    "什么是EPSG代码",
    "中国常用的坐标系有哪些",

    # 数据模型
    "什么是矢量数据",
    "什么是栅格数据",
    "矢量数据和栅格数据的区别",
    "什么是TIN",
    "什么是DEM",
    "什么是DSM",
    "什么是DTM",
    "DEM和DSM的区别是什么",
    "Shapefile是什么格式",
    "GeoJSON是什么",
    "GeoPackage是什么",
    "GeoTIFF是什么",
    "KML和KMZ的区别",
    "什么是空间索引",
    "什么是拓扑关系",
    "CAD数据和GIS数据的区别",

    # GPS/GNSS
    "GPS和北斗的区别",
    "什么是GNSS",
    "北斗卫星导航系统是什么",
    "什么是差分GPS",
    "什么是RTK定位",
    "GPS的工作原理是什么",
    "什么是多路径效应",
]

# ===== GIS 操作类 (gis_operation) — RAG 检索 =====
gis_operation = [
    # QGIS 相关
    "QGIS怎么安装",
    "QGIS怎么导入shapefile",
    "QGIS怎么添加底图",
    "QGIS怎么做缓冲区分析",
    "QGIS怎么修改投影",
    "QGIS怎么做专题图",
    "QGIS怎么添加标注",
    "QGIS怎么导出地图",
    "QGIS怎么安装插件",
    "QGIS怎么做空间查询",
    "QGIS怎么做字段计算",
    "QGIS怎么裁剪栅格",
    "QGIS怎么合并矢量图层",
    "QGIS怎么做坡度分析",
    "QGIS怎么做热力图",
    "QGIS怎么导入GPS数据",
    "QGIS怎么连接PostGIS",
    "QGIS怎么设置样式",
    "QGIS怎么做叠加分析",
    "QGIS怎么导出为GeoJSON",
    "QGIS怎么把坐标点转成面",
    "QGIS怎么计算面积",
    "QGIS怎么做网络分析",
    "QGIS中怎么批量处理文件",
    "QGIS怎么制作三维地图",
    "QGIS怎么导入在线地图服务",
    "QGIS怎么使用Python脚本",

    # ArcGIS 相关
    "ArcGIS怎么导入数据",
    "ArcGIS怎么进行空间校正",
    "ArcGIS怎么创建要素类",
    "ArcGIS怎么做叠加分析",
    "ArcGIS怎么导出地图",
    "ArcGIS怎么创建地理数据库",
    "ArcMap和ArcGIS Pro有什么区别",
    "ArcGIS怎么设置坐标系",
    "ArcGIS怎么做缓冲区",
    "ArcGIS怎么使用ModelBuilder",

    # 数据处理
    "怎么将CAD数据转成GIS格式",
    "怎么将表格数据导入GIS",
    "怎么给Shapefile添加字段",
    "怎么合并多个Shapefile",
    "怎么做地图配准",
    "怎么数字化纸质地图",
    "怎么做空间插值",
    "怎么创建等高线",
    "怎么从DEM中提取流域",
    "怎么做视域分析",
    "怎么生成坡度图",
    "怎么做土地利用分类",
    "怎么用遥感影像做变化检测",

    # 软件操作
    "ENVI怎么打开遥感影像",
    "ERDAS怎么进行监督分类",
    "ArcGIS中怎么进行网络分析",
    "如何在QGIS中使用OSM数据",
    "怎样在GIS中进行地址匹配",
    "如何使用Google Earth Engine",
    "如何用Python处理Shapefile",
    "如何使用GDAL库处理栅格数据",
    "怎么用GeoPandas做空间分析",
]

# ===== 闲聊/其他 (general) — LLM 原生回答 =====
general = [
    "你好",
    "你是谁",
    "今天天气怎么样",
    "谢谢",
    "你能做什么",
    "你叫什么名字",
    "再见",
    "早上好",
    "晚安",
    "帮我查一下",
    "帮我分析一下",
    "你好吗",
    "请介绍一下自己",
    "你有什么功能",
    "很晚了该休息了",
    "在吗",
    "你好呀",
    "你了解哪些内容",
    "我有个问题",
    "帮帮我",
    "你擅长什么",
]

# ===== 写入 CSV =====
all_data = []

for text in gis_term:
    all_data.append((text, "gis_term"))
for text in gis_operation:
    all_data.append((text, "gis_operation"))
for text in general:
    all_data.append((text, "general"))

random.shuffle(all_data)

os.makedirs("data/splits", exist_ok=True)

with open("data/splits/router_train.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    # 80% 训练
    train_end = int(len(all_data) * 0.8)
    for text, label in all_data[:train_end]:
        writer.writerow([text, label])

print(f"Router training data generated:")
print(f"  gis_term:      {len(gis_term)} samples")
print(f"  gis_operation: {len(gis_operation)} samples")
print(f"  general:       {len(general)} samples")
print(f"  Total train:   {train_end}")
print(f"  Saved to:      data/splits/router_train.csv")
