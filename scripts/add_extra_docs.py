"""Add remaining documents to reach 100+"""
import json

with open("projects/geoai-assistant/knowledge_base/demo_docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

extra = [
    {
        "id": "gis-134",
        "title": "ArcGIS Pro和ArcMap的主要区别",
        "category": "arcgis_ops",
        "content": "ArcGIS Pro是ESRI新一代桌面GIS平台（2015年发布），逐步取代经典的ArcMap。主要区别：（1）架构 — ArcMap是32位单线程应用（受内存限制），ArcGIS Pro是64位多线程应用，处理大数据效率提升显著；（2）界面 — Pro采用Ribbon界面（类似Office），ArcMap使用传统菜单+工具栏；（3）二维三维 — Pro原生支持三维场景，可在同一工程中同时使用地图视图和场景视图，ArcMap需要ArcScene/ArcGlobe扩展；（4）工程管理 — Pro使用.aprx工程文件统一管理地图、布局、工具箱等，ArcMap使用.mxd文档；（5）Python — Pro使用Python 3.x + arcpy，ArcMap使用Python 2.7（已停止维护）；（6）许可 — Pro支持Named User许可（通过ArcGIS Online/Portal账号），支持离线授权。ESRI已宣布ArcMap 10.8.x为最后版本，建议新项目全部迁移到ArcGIS Pro。",
        "keywords": ["ArcGIS Pro", "ArcMap", "64位", "Ribbon", "3D", "APRX", "升级"]
    },
    {
        "id": "gis-135",
        "title": "ArcGIS中的属性选择与空间选择",
        "category": "arcgis_ops",
        "content": "在ArcGIS中进行数据选择和查询的方法：（1）按属性选择 —「选择→按属性选择」，使用SQL WHERE子句构建查询。常用语法：字符串条件 NAME = 'Suzhou'，数字条件 POPULATION > 500000，组合用AND/OR，模糊匹配用LIKE（如 NAME LIKE '%州%'）；（2）按位置选择 —「选择→按位置选择」，基于空间关系筛选要素。关系类型：相交、包含、完全包含、在一定距离内、共享线段等，如选择某行政区内的所有POI点；（3）选择集操作 — 支持添加到当前选择、从选择中移除、在当前选择中选择、新建选择四种模式；（4）导出选择集 — 右键图层→「数据→导出数据」，可选择只导出选中要素；（5）定义查询（Definition Query）— 在图层属性中设置SQL过滤条件，使图层始终只显示符合条件的要素而无需删除数据，适合大型数据集的交互式筛选。",
        "keywords": ["ArcGIS选择", "SQL查询", "空间选择", "定义查询", "按属性选择"]
    },
    {
        "id": "gis-136",
        "title": "GIS在智慧城市中的应用",
        "category": "common_faq",
        "content": "GIS是智慧城市建设的核心基础设施之一：（1）城市三维建模与数字孪生 — 利用倾斜摄影、LiDAR点云、BIM数据构建城市三维模型，支持城市场景的可视化和管理模拟；（2）地下管线综合管理 — 将给排水、燃气、电力、通信等管线纳入统一GIS平台，实现碰撞检测和开挖分析；（3）城市规划 — 利用多源空间数据支持用地适宜性评价、交通可达性分析、公共服务设施布局优化等决策；（4）应急指挥 — 结合IoT传感器数据进行灾害预警、应急资源调度和疏散路径规划；（5）生态环境监测 — 利用遥感和GIS监测城市热岛效应、绿地变化、空气质量分布；（6）交通管理 — 拥堵热点分析、公交线路优化、停车位管理。主流技术架构：PostGIS/SDE（数据存储）→ GeoServer/ArcGIS Server（服务发布）→ Cesium.js/Mapbox GL JS（前端展示）。",
        "keywords": ["智慧城市", "数字孪生", "地下管线", "城市三维", "应急指挥", "物联网"]
    },
    {
        "id": "gis-137",
        "title": "GIS和BIM的集成应用",
        "category": "common_faq",
        "content": "BIM（Building Information Modeling）和GIS的集成正在成为基础设施全生命周期管理的关键技术。GIS管理宏观地理环境（室外），BIM管理微观建筑细节（室内构件级）。集成方法：（1）数据格式转换 — 将BIM模型（IFC/RVT格式）转为GIS可读格式。ArcGIS Pro支持直接读取Revit(.rvt)文件并转为3D要素；QGIS可通过IFC插件导入IFC文件；（2）坐标配准 — BIM使用局部施工坐标系，需转换到真实地理坐标（CGCS2000/WGS84）才能与GIS叠加。通常在BIM建模时设置正确的测量点和基准点；（3）应用场景 — 城市规划审批（将BIM模型放入城市三维GIS环境审查天际线、日照、景观影响）、资产管理（建筑构件级别属性挂接到地理信息）、室内外导航无缝衔接。Autodesk和ESRI已达成战略合作，ArcGIS Pro和InfraWorks可原生读取Revit模型。",
        "keywords": ["BIM", "GIS+BIM", "IFC", "Revit", "三维配准", "建筑信息模型"]
    },
    {
        "id": "gis-138",
        "title": "QGIS制作三维地形图的方法",
        "category": "qgis_ops",
        "content": "在QGIS中创建三维地形可视化的方法：（1）使用QGIS2threejs插件 —「插件→管理和安装插件」搜索安装QGIS2threejs。打开插件后选择DEM图层作为地形高程源，设置垂直夸张系数（通常1.5-3倍），将其他图层（影像、矢量）作为覆盖层叠加在地形上，点击「导出」可生成交互式HTML三维场景（基于Three.js），在浏览器中可旋转缩放浏览；（2）使用原生3D地图视图 — QGIS 3.x内置了3D地图视图（「视图→新建3D地图视图」），在配置对话框中设置地形数据和垂直缩放，可将2D图层叠加到3D表面；（3）输出 — QGIS2threejs可导出HTML（适合分享）或PNG截图，3D地图视图支持右键导出为图片。三维可视化对地形分析、景观规划、项目汇报非常有用。如需更专业的三维效果（飞行动画），可考虑Blender GIS插件或ArcGIS Pro三维场景。",
        "keywords": ["QGIS三维", "QGIS2threejs", "3D地图", "地形可视化", "Three.js"]
    },
    {
        "id": "gis-139",
        "title": "QGIS中使用表达式的高级技巧",
        "category": "qgis_ops",
        "content": "QGIS表达式引擎功能强大：（1）条件样式 — 在图层符号化中选择「基于规则」，设多条规则：如面积大于10000用深色，1000-10000用中色，小于1000用浅色；（2）数据驱动覆盖 — 符号几乎所有属性（大小、颜色、旋转角度）都可点击右侧表达式图标动态控制。例如用人口值控制点符号大小、用方向字段控制箭头旋转；（3）多行标注 — 如 CONCAT(城市名, '\n人口:', FORMAT_NUMBER(人口,0), '万') 实现分两行标注字段；（4）条件标注 — 用 CASE WHEN 条件=1 THEN 字段名 END 控制只标注符合条件的要素；（5）虚拟字段 — 图层属性→字段→新建虚拟字段，使用表达式定义动态字段如 $area（实时计算面积），随几何变化自动更新。表达式函数参考可在表达式构建器左侧面板搜索浏览。",
        "keywords": ["QGIS表达式", "数据驱动", "条件样式", "虚拟字段", "标注表达式"]
    },
    {
        "id": "gis-140",
        "title": "KML/KMZ数据格式详解",
        "category": "data_formats",
        "content": "KML（Keyhole Markup Language）是由Google开发的基于XML的地理数据格式，广泛应用于Google Earth/Google Maps。KMZ是KML的压缩版本（ZIP格式，包含doc.kml主文件和资源文件夹如images/、models/等）。（1）KML要素类型 — Placemark（点，支持自定义图标）、LineString（线）、Polygon（面）、GroundOverlay（地面叠加图片）、NetworkLink（动态链接远程KML，实现实时数据更新）；（2）使用场景 — KML适合在Google Earth中查看标记、轨迹、分布图，以及向公众分发简单地理数据（无需专业GIS软件）；（3）局限 — KML无空间索引、属性表简单（HTML弹出窗口）、只支持WGS84坐标系，不适合完整GIS分析。QGIS中导入导出在「矢量」菜单；ArcGIS中使用「转换工具→由KML转出」/「转为KML」。KMZ可用7-Zip解压后单独提取KML编辑。",
        "keywords": ["KML", "KMZ", "Google Earth", "XML", "Placemark", "数据交换"]
    },
    {
        "id": "gis-141",
        "title": "NDWI水体指数与实际应用",
        "category": "remote_sensing",
        "content": "NDWI（Normalized Difference Water Index，归一化水体指数）用于识别和提取地表水体。两个常用版本：（1）McFeeters(1996) NDWI = (Green - NIR) / (Green + NIR)。水体在绿光波段有一定反射、近红外波段强烈吸收，因此水体区域NDWI为正。Landsat 8 对应 (Band3 - Band5) / (Band3 + Band5)；（2）MNDWI（改进型）= (Green - SWIR1) / (Green + SWIR1)，在城市区域抑制建筑背景噪声效果更好。Landsat 8 对应 (Band3 - Band6) / (Band3 + Band6)；（3）阈值提取 — NDWI > 0 通常可区分水体，最优阈值因影像条件而异，建议结合Otsu自动阈值法微调；（4）应用 — 湖泊面积动态监测、洪涝灾害范围评估（洪水前后NDWI差异）、海岸线变化分析、湿地保护。QGIS中用栅格计算器执行NDWI运算：对于Sentinel-2，用Band3(Green)和Band8(NIR)。",
        "keywords": ["NDWI", "水体指数", "水体提取", "MNDWI", "洪涝监测"]
    },
    {
        "id": "gis-142",
        "title": "获取最新免费高清卫星影像的渠道",
        "category": "common_faq",
        "content": "获取最新免费高清卫星影像的主要渠道：（1）Sentinel-2（ESA）— 10米分辨率，每5天重访全球覆盖免费。可通过ESA Copernicus Data Space Ecosystem的WMS/WMTS在线服务直接在QGIS中加载而无需下载；（2）Landsat 8/9（USGS）— 30米多光谱+15米全色，每16天重访。全色锐化后可到15米，历史存档可追溯到1972年；（3）PlanetScope — 3-5米分辨率每日全球覆盖，需付费订阅或教育研究计划申请免费使用；（4）Google/Bing底图 — 亚米级高清影像覆盖城市区域，只能通过XYZ Tiles加载为底图无法下载分析。QGIS中通过QuickMapServices插件添加；（5）中国高分系列 — 亚米级数据通过自然资源卫星遥感云服务平台（sasclouds.com）查询预订；（6）Maxar/WorldView — 0.3-0.5米超高分辨率商用付费，灾区和人道主义地区可通过Maxar Open Data Program免费获取。建议：高频监测用Sentinel-2，高精度用付费商业卫星，历史分析用Landsat，城市底图直接XYZ Tiles。",
        "keywords": ["卫星影像下载", "Sentinel-2", "Planet", "Maxar", "高分卫星", "底图"]
    },
]

docs.extend(extra)
with open("projects/geoai-assistant/knowledge_base/demo_docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

cats = {}
for d in docs:
    cats[d["category"]] = cats.get(d["category"], 0) + 1
print(f"\nTotal: {len(docs)} docs\n")
for k, v in sorted(cats.items()):
    print(f"  {k}: {v}")
