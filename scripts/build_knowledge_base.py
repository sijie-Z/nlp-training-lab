#!/usr/bin/env python3
"""
修复 + 扩展 GeoAI Assistant 知识库
修复后台 agent 写坏的 JSON（ASCII 双引号导致语法错误）
并补充未完成的类别
"""
import json
import os

KB_PATH = "projects/geoai-assistant/knowledge_base/demo_docs.json"

# 先尝试加载已有文件
existing_docs = []
try:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        existing_docs = json.load(f)
    print(f"Loaded {len(existing_docs)} existing docs")
except json.JSONDecodeError:
    print("Existing JSON is broken, rebuilding from scratch...")
    # Try to salvage partial docs by reading lines
    with open(KB_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    # 找到所有完整的对象（在 `},` 或 `}` 处切割）
    # 手动提取还能用的文档
    import re
    # 匹配每个完整的 { ... } 对象
    pattern = r'\{\s*"id":\s*"([^"]+)",\s*"title":\s*"([^"]+)",\s*"category":\s*"([^"]+)",\s*"content":\s*"([^"]*)",\s*"keywords":\s*(\[[^\]]*\])'
    matches = re.findall(pattern, content, re.DOTALL)
    for m in matches:
        try:
            kw = json.loads(m[4])
            existing_docs.append({
                "id": m[0],
                "title": m[1],
                "category": m[2],
                "content": m[3],
                "keywords": kw
            })
        except:
            pass
    print(f"Salvaged {len(existing_docs)} docs from broken JSON")

# 已有的 ID 集合
existing_ids = {d["id"] for d in existing_docs}

# ===== 新增文档（补充到 100+） =====
new_docs = [
    # ===== 遥感补充 (15条目标) =====
    {
        "id": "gis-105",
        "title": "遥感影像的波段组合",
        "category": "remote_sensing",
        "content": "遥感影像由多个波段组成，每个波段记录特定波长范围的电磁辐射。Landsat 8 OLI传感器有11个波段：波段1（海岸/气溶胶，0.43-0.45μm）、波段2（蓝光，0.45-0.51μm）、波段3（绿光，0.53-0.59μm）、波段4（红光，0.64-0.67μm）、波段5（近红外，0.85-0.88μm）、波段6（短波红外1，1.57-1.65μm）、波段7（短波红外2，2.11-2.29μm）、波段8（全色，0.50-0.68μm）、波段9（卷云，1.36-1.38μm）、波段10（热红外1，10.6-11.19μm）、波段11（热红外2，11.5-12.51μm）。常用的假彩色组合：波段543（近红外+红+绿）适合植被分析；波段764（短波红外+近红外+红）适合地质分析；波段432（红+绿+蓝）为真彩色，适合目视解译。",
        "keywords": ["波段组合", "Landsat8", "假彩色", "OLI", "近红外", "真彩色", "波段"]
    },
    {
        "id": "gis-106",
        "title": "遥感影像的大气校正方法",
        "category": "remote_sensing",
        "content": "大气校正的目的是消除大气散射、吸收对遥感影像的影响，获取地表真实反射率。常用方法包括：（1）暗目标法（DOS, Dark Object Subtraction）——假设影像中存在阴影或深水等反射率接近零的区域，将这些区域的DN值作为大气程辐射的估计值进行扣除，适合快速处理；（2）6S模型（Second Simulation of the Satellite Signal in the Solar Spectrum）——基于辐射传输方程进行物理校正，精度高但需要输入大气参数（气溶胶光学厚度、水汽含量等）；（3）FLAASH（Fast Line-of-sight Atmospheric Analysis of Spectral Hypercubes）——ENVI中常用的商业大气校正模块，基于MODTRAN4辐射传输模型；（4）ATCOR——用于平坦地形和山地地形的大气/地形校正，支持大多数卫星传感器。对于定量遥感分析（如植被指数计算、变化检测），大气校正通常是一个必要步骤。",
        "keywords": ["大气校正", "DOS", "6S模型", "FLAASH", "ATCOR", "暗目标法", "辐射传输"]
    },
    {
        "id": "gis-107",
        "title": "常用的遥感卫星数据源",
        "category": "remote_sensing",
        "content": "获取免费遥感影像的主要渠道：（1）USGS EarthExplorer（earthexplorer.usgs.gov）——Landsat系列、ASTER、MODIS、SRTM DEM等，全球覆盖，历史存档完整；（2）ESA Copernicus Open Access Hub（scihub.copernicus.eu）——Sentinel-1（SAR）、Sentinel-2（多光谱）、Sentinel-3（海洋/陆地监测）等；（3）NASA Earthdata Search——MODIS产品、 VIIRS等；（4）中国资源卫星应用中心——高分系列（GF-1至GF-7）、资源系列等国产卫星数据，需注册申请；（5）Google Earth Engine——在线平台提供Landsat、Sentinel、MODIS等数据的在线访问和处理，无需下载即可分析；（6）地理空间数据云（gscloud.cn）——中国科学院提供的国产和国外卫星数据镜像。选择数据时需考虑空间分辨率、时间分辨率、光谱分辨率和数据获取成本等因素。",
        "keywords": ["卫星数据下载", "USGS", "EarthExplorer", "Copernicus", "Sentinel", "Landsat", "高分卫星"]
    },
    {
        "id": "gis-108",
        "title": "遥感影像的土地利用分类方法",
        "category": "remote_sensing",
        "content": "土地利用/土地覆盖分类（LULC）是遥感的核心应用之一。主要方法：（1）监督分类——需要人工选取训练样本，常用算法包括最大似然分类（MLC）、支持向量机（SVM）、随机森林（RF）、卷积神经网络（CNN）。步骤为：定义分类体系→选取训练样本→选择算法→分类→精度评价；（2）非监督分类——无需训练样本，由算法自动聚类，如ISODATA、K-Means。然后人工赋予类别含义；（3）面向对象分类（OBIA）——先对影像分割为对象（如eCognition软件），再基于对象的纹理、形状、光谱等特征分类，适合高分辨率影像；（4）深度学习分类——使用U-Net、DeepLab等语义分割网络进行像素级分类，精度最高但需要大量标注数据。精度评价常用混淆矩阵、总体精度（OA）和Kappa系数。",
        "keywords": ["土地利用分类", "监督分类", "非监督分类", "面向对象分类", "OBIA", "随机森林", "混淆矩阵"]
    },
    {
        "id": "gis-109",
        "title": "植被指数NDVI详解",
        "category": "remote_sensing",
        "content": "NDVI（Normalized Difference Vegetation Index，归一化植被指数）是最常用的植被指数，计算公式为：NDVI = (NIR - Red) / (NIR + Red)，其中NIR为近红外波段反射率，Red为红光波段反射率。NDVI取值范围为[-1, 1]，通常绿色植被在0.2-0.8之间。NDVI的原理基于植被的光谱特性：健康植被在近红外波段强烈反射，在红光波段被叶绿素强烈吸收。因此植被越茂盛，NDVI值越高。负值通常代表水体，接近0代表裸土或建筑。NDVI的应用包括：作物长势监测、干旱评估、植被覆盖度估算、物候分析等。类似指数还有EVI（增强型植被指数）、SAVI（土壤调节植被指数）、NDWI（归一化水体指数）等。",
        "keywords": ["NDVI", "归一化植被指数", "植被指数", "近红外", "红光", "植被覆盖度", "EVI"]
    },
    {
        "id": "gis-110",
        "title": "SAR雷达遥感简介",
        "category": "remote_sensing",
        "content": "SAR（Synthetic Aperture Radar，合成孔径雷达）是一种主动微波遥感技术，通过发射微波脉冲并接收地面回波来成像。SAR的主要优势：（1）全天候——不受云雨雾等天气影响；（2）全天时——白天黑夜均可成像；（3）对地物几何结构敏感，可以发现地表微小形变。SAR的关键参数包括：波长（波段）——X波段（3cm）、C波段（5.6cm，如Sentinel-1）、L波段（23.5cm，如ALOS-2），波长越长穿透力越强；极化方式——HH、VV、HV、VH，不同极化对地物敏感度不同。InSAR（干涉SAR）通过比较两幅SAR影像的相位差，可测量地表形变（精度达毫米级），广泛应用于地震形变监测、地面沉降监测、火山活动监测等领域。DInSAR（差分干涉）和PS-InSAR（永久散射体干涉）是更高级的形变监测技术。",
        "keywords": ["SAR", "合成孔径雷达", "InSAR", "雷达遥感", "微波", "形变监测", "Sentinel-1"]
    },

    # ===== 坐标系统补充 =====
    {
        "id": "gis-111",
        "title": "中国常用坐标系详解",
        "category": "coordinate_systems",
        "content": "中国常用的坐标系包括：（1）CGCS2000（2000国家大地坐标系）——中国现行的法定坐标系，2008年启用，基于ITRF97框架，历元2000.0。椭球参数与WGS84非常接近但略有差异；（2）WGS84——GPS使用的全球坐标系，Google Maps、百度地图（加密后）等在线地图使用；（3）北京54（BJ54）——1954年建立的参心坐标系，基于克拉索夫斯基椭球，已逐步淘汰但大量历史数据仍使用；（4）西安80（XA80）——1980年建立的参心坐标系，基于IAG75椭球，精度高于BJ54但仍为参心系。坐标转换注意事项：不同坐标系之间的转换需要七参数（三个平移、三个旋转、一个尺度因子）或四参数（两个平移、一个旋转、一个尺度，适用于小范围）。实际工作中，CGCS2000和WGS84在小比例尺下可近似等同，但在大比例尺（>1:10000）时需要使用正式的坐标转换参数。中国的大地测量基准中，高程系统使用1985国家高程基准。",
        "keywords": ["CGCS2000", "北京54", "西安80", "WGS84", "七参数", "坐标转换", "大地测量"]
    },
    {
        "id": "gis-112",
        "title": "高斯-克吕格投影详解",
        "category": "coordinate_systems",
        "content": "高斯-克吕格（Gauss-Kruger）投影是中国大比例尺地形图的标准投影。它是一种横轴等角切椭圆柱投影——想象一个椭圆柱横切地球的某条子午线（中央经线），将两侧各3度（或6度）的区域投影到圆柱面上。关键参数：（1）分带方式——6度带（1:2.5万-1:50万地形图）全球分60带，中国跨13-23带；3度带（1:1万及以上大比例尺）全球分120带；（2）中央经线——6度带L0=6n-3，3度带L0=3n（n为带号）；（3）坐标加常数——为消除负值，X坐标（北方向）加0，Y坐标（东方向）加500000米（500公里），并在Y值前冠以带号；（4）投影变形——中央经线上无长度变形，离中央经线越远变形越大，6度带边缘最大变形约1/1000。UTM（通用横轴墨卡托）投影与高斯投影原理相似但参数不同（UTM为割圆柱投影，中央经线比例因子为0.9996），两者不可混用。",
        "keywords": ["高斯克吕格投影", "高斯投影", "6度带", "3度带", "中央经线", "投影分带", "加常数"]
    },
    {
        "id": "gis-113",
        "title": "如何为项目选择合适的投影",
        "category": "coordinate_systems",
        "content": "选择投影坐标系的原则：（1）考虑地理位置——中国范围内优先使用CGCS2000坐标系的Albers等积圆锥投影（全国小比例尺图）或高斯-克吕格投影（大比例尺图）；（2）考虑分析目的——面积计算使用等积投影（如Albers）；距离和方向使用等距投影；导航使用等角投影（如墨卡托）；（3）考虑比例尺——1:100万以下用小比例尺Albers/Lambert圆锥投影；1:1万-1:100万用6度带高斯投影；1:500-1:5000用3度带高斯投影；（4）QGIS操作：「右键图层→导出→另存为」选择目标CRS可以重投影；或使用「栅格→投影→重投影」/「矢量→数据管理工具→重投影图层」批量处理。不确定时，检查同区域已有权威数据的投影并保持一致。中国省级行政区图常用Albers等积圆锥投影（双标准纬线25°N和47°N，中央经线105°E）。",
        "keywords": ["投影选择", "Albers投影", "等积投影", "比例尺", "投影参数", "重投影"]
    },

    # ===== QGIS 操作补充 =====
    {
        "id": "gis-114",
        "title": "QGIS制作专题地图的完整步骤",
        "category": "qgis_ops",
        "content": "在QGIS中制作出版级专题地图的完整流程：（1）准备数据和图层，确保所有图层使用正确的坐标系；（2）设置图层样式和符号化——右键图层→属性→符号化，根据属性字段设置分类/分级颜色；（3）添加标注——图层属性→标注，选择标注字段，设置字体、大小、位置和避让规则；（4）新建打印布局——「项目→新建打印布局」，给布局命名；（5）添加地图——「添加项目→添加地图」，在地图项属性中设置比例尺；（6）添加地图元素——图例（自动读取图层名和符号）、指北针、比例尺条、标题文本；（7）添加网格/经纬网——地图项属性→网格，设置间隔和样式；（8）导出——「布局→导出为图片/PDF/SVG」，分辨率建议300dpi用于印刷。QGIS自带丰富的符号库（颜色渐变Fill如Blues、Reds、Viridis等），也可以用SVG符号和自定义颜色。打印布局支持多地图对比（如主图+位置索引图）。",
        "keywords": ["QGIS专题图", "打印布局", "符号化", "图例", "指北针", "比例尺", "地图输出"]
    },
    {
        "id": "gis-115",
        "title": "QGIS常用插件推荐",
        "category": "qgis_ops",
        "content": "QGIS的强大之处很大程度上来自其插件生态，以下是最常用的插件推荐：（1）QuickMapServices——一键添加Google Maps、OSM、Bing等在线底图，是使用最频繁的插件；（2）Semi-Automatic Classification Plugin (SCP)——遥感影像半自动分类，支持Landsat、Sentinel预处理和分类；（3）qgis2web——将QGIS项目一键导出为Leaflet或OpenLayers交互式Web地图；（4）MMQGIS——地理编码、缓冲区合并、属性连接等实用工具集合；（5）Profile Tool——生成地形剖面图；（6）FlowMapper——流向图和OD分析；（7）TimeManager——时间序列动画展示；（8）QuickOSM——在线下载OpenStreetMap特定要素（如某区域的医院、学校）；（9）Lat Lon Tools——坐标格式转换和坐标点定位。安装方法：「插件→管理和安装插件」，搜索名称后点击安装。",
        "keywords": ["QGIS插件", "QuickMapServices", "SCP", "qgis2web", "MMQGIS", "QuickOSM", "插件推荐"]
    },
    {
        "id": "gis-116",
        "title": "QGIS中属性表的常用操作",
        "category": "qgis_ops",
        "content": "QGIS属性表操作是矢量数据分析的基础：（1）打开属性表——右键图层→打开属性表，或按F6；（2）选择要素——可按属性选择（「按表达式选择」按钮，如 field='value'）、按位置选择、手动画框选择；（3）字段计算器——「打开字段计算器」按钮，可创建新字段或更新已有字段，常用函数包括$area（计算面积平方米）、$length（计算长度米）、$x/$y（获取质心坐标）、round()、concat()等；（4）字段编辑——切换编辑模式后才能修改字段值；添加/删除字段需在「图层属性→字段」中操作；（5）表连接（Join）——图层属性→连接，基于共同字段将外部表格（CSV/Excel）关联到属性表，注意连接字段类型和编码一致；（6）统计——「显示统计面板」可查看选中要素的计数、求和、均值、标准差等。QGIS的字段计算器表达式功能非常强大，支持条件语句（CASE WHEN）、几何函数和字符串操作。",
        "keywords": ["QGIS属性表", "字段计算器", "表达式选择", "表连接", "属性编辑", "字段计算"]
    },
    {
        "id": "gis-117",
        "title": "QGIS中进行空间分析的方法",
        "category": "qgis_ops",
        "content": "QGIS内置了丰富的空间分析工具，无需ArcGIS即可完成大多数分析任务：（1）缓冲区分析——「矢量→地理处理工具→缓冲区」，输入图层和距离，可选择融合重叠缓冲区和端头样式；（2）叠加分析——「矢量→地理处理工具」下有裁剪、差集、交集、并集、对称差等工具，如用行政区裁剪路网；（3）空间连接——「矢量→数据管理工具→按位置连接属性」，如将每个POI点关联所在行政区名称；（4）栅格分析——「栅格→地形分析」提供坡度、坡向、山体阴影、可见度等工具；「栅格→提取」可按掩膜裁剪栅格；（5）数据管理——「矢量→数据管理工具→按位置合并」、「按属性合并」用于矢量数据融合合并；（6）网络分析——使用Road Graph插件或QNEAT3插件进行最短路径和服务区分析。所有工具支持批处理模式（右键工具→作为批处理执行），可对多个文件执行相同操作。",
        "keywords": ["QGIS空间分析", "缓冲区", "叠加分析", "空间连接", "栅格分析", "批处理"]
    },

    # ===== ArcGIS 操作补充 =====
    {
        "id": "gis-118",
        "title": "ArcGIS中创建地理数据库",
        "category": "arcgis_ops",
        "content": "在ArcGIS中创建地理数据库（Geodatabase）的步骤：（1）在Catalog窗口中，右键目标文件夹→「新建→文件地理数据库（.gdb）」，命名后即可创建FGDB；（2）在新建的.gdb上右键→「新建→要素数据集」，设置空间参考（坐标系），并设置XY容差和Z容差；（3）在要素数据集上右键→「新建→要素类」，定义字段（名称、类型、长度），可选择是否启用M值（线性参考）和Z值（3D）；（4）创建拓扑规则——要素数据集右键→「新建→拓扑」，选择参与的要素类并设置拓扑规则（如不能重叠、不能有空隙、必须被其他要素覆盖等）；（5）创建关系类——用于定义不同要素类之间的关联关系（一对一、一对多）；（6）属性域和子类型——在.gdb属性中创建属性域（范围域或编码值域），然后为要素类的特定字段分配属性域，以约束输入值。FGDB相比Shapefile的优势：支持长字段名、支持拓扑、更大的存储容量（理论上无限）、更好的并发访问性能。企业级应用建议使用ArcGIS Enterprise Geodatabase（基于SQL Server/Oracle/PostgreSQL + ArcSDE）。",
        "keywords": ["ArcGIS地理数据库", "FGDB", "要素数据集", "拓扑规则", "属性域", "子类型"]
    },
    {
        "id": "gis-119",
        "title": "ArcGIS中的符号化与地图制作",
        "category": "arcgis_ops",
        "content": "ArcGIS Pro/ArcMap中进行符号化和地图制作的流程：（1）符号系统——图层属性→符号系统，可选单一符号、类别（唯一值）、数量（分级色彩/分级符号）、图表（饼图/柱图）、多个属性等；（2）符号选择器——从样式库中选择或自定义符号，可导入图片作为点符号；（3）标注——图层属性→标注，设置标注字段、字体、放置位置和权重等级；使用标注表达式（Python/VBScript）可实现复杂标注如多字段合并；（4）布局视图——插入图例、指北针、比例尺、图名、数据框（鹰眼图/位置索引图）；（5）数据驱动页面（Data Driven Pages）——按格网自动生成系列地图，适合带状图或分幅输出；（6）导出——「共享→导出布局」选择格式（PDF/TIFF/JPEG/PNG等），设置分辨率和压缩参数。ArcGIS自带的ESRI样式库包含大量专业地图符号，也支持导入.style文件。符号级别绘制可以控制多符号图层的绘制顺序和连接效果。",
        "keywords": ["ArcGIS符号化", "符号系统", "数据驱动页面", "布局", "ESRI样式", "标注表达式"]
    },
    {
        "id": "gis-120",
        "title": "ArcGIS的ModelBuilder自动化建模",
        "category": "arcgis_ops",
        "content": "ModelBuilder是ArcGIS内置的可视化工作流建模工具，通过拖拽方式将多个GP工具串联为自动化处理流程：（1）打开ModelBuilder——「分析→ModelBuilder」或ArcToolbox右键→新建模型；（2）基本元素——蓝色椭圆=输入数据，黄色矩形=工具/处理步骤，绿色椭圆=输出数据；（3）连接——从数据拖线到工具设置输入，从工具拖线到输出数据创建结果；（4）运行——右键「运行」单次执行，或右键「批量」对多个输入自动循环；（5）迭代器——「插入→迭代器」，如迭代要素类（遍历.gdb中的所有要素）、迭代栅格、迭代字段值等，实现批量自动化；（6）模型参数——将变量设为「模型参数」（变量右键→模型参数），则运行模型时可交互选择输入输出路径；（7）导出为Python——「导出→导出为Python脚本」，可将模型转为arcpy代码，便于进一步定制或嵌入其他程序。ModelBuilder特别适合数据处理流水线（如：数据导入→投影变换→裁剪→属性计算→符号化输出），能大幅减少重复操作。",
        "keywords": ["ModelBuilder", "ArcGIS模型", "空间建模", "迭代器", "自动化", "arcpy", "可视化建模"]
    },

    # ===== FAQ 补充 =====
    {
        "id": "gis-121",
        "title": "学GIS需要什么编程基础",
        "category": "common_faq",
        "content": "GIS学习是否需要编程取决于你的职业方向：（1）GIS数据分析师/制图员——不强制需要编程，但会SQL和Python会大大提高效率，主要使用QGIS/ArcGIS的图形界面操作；（2）GIS开发工程师——必须掌握至少一门编程语言。Python是GIS领域最常用的语言（Arcpy、GeoPandas、GDAL、PyQGIS等库），JavaScript用于Web GIS开发（Leaflet、OpenLayers、Mapbox GL JS、Cesium），SQL用于空间数据库操作（PostGIS、MySQL Spatial）；（3）遥感分析——推荐Python（GDAL、Rasterio、scikit-learn、TensorFlow/PyTorch）或GEE的JavaScript API；（4）初学者的学习路线建议：先掌握GUI操作理解GIS核心概念→再学Python做自动化→根据需求学习Web开发或空间数据库。Python入门推荐从GeoPandas开始，它可以像操作Excel表格一样处理空间数据，学习曲线很低。",
        "keywords": ["GIS学习", "Python GIS", "编程基础", "GeoPandas", "GIS开发", "入门"]
    },
    {
        "id": "gis-122",
        "title": "免费GIS软件推荐",
        "category": "common_faq",
        "content": "以下是最实用的免费/开源GIS软件推荐：（1）QGIS——功能最全面的开源桌面GIS，媲美ArcGIS，支持丰富的插件生态，完全免费，有活跃的中文社区；（2）Google Earth Pro——免费的三维虚拟地球，适合快速浏览和简单测量；（3）GeoDa——专注于空间统计分析和探索性空间数据分析（ESDA），如空间自相关（Moran's I）、LISA聚集图等；（4）SAGA GIS——德国开发的专注于栅格分析和地形分析的GIS，有强大的地统计和地形分析工具；（5）GRASS GIS——历史最悠久的开源GIS（1982年开始），在栅格/影像处理方面能力突出，可通过QGIS调用其工具；（6）uDig——适合只想查看和简单编辑GIS数据的轻量用户；（7）PostGIS——开源空间数据库，在PostgreSQL上扩展空间能力，适合管理大规模空间数据；（8）Google Earth Engine——在线遥感大数据分析平台，免费用于教育和研究。对于大多数日常GIS工作，QGIS + PostGIS组合即可满足需求。",
        "keywords": ["免费GIS", "开源GIS", "QGIS", "GeoDa", "SAGA", "GRASS", "PostGIS"]
    },
    {
        "id": "gis-123",
        "title": "GIS数据从哪里获取",
        "category": "common_faq",
        "content": "常用的免费GIS数据获取渠道：（1）OpenStreetMap（OSM）——全球开源地图数据，包含道路、建筑、POI等丰富要素，可通过QGIS的QuickOSM插件下载或Geofabrik下载按国家/地区的导出包；（2）全国地理信息资源目录服务系统（webmap.cn）——提供1:100万、1:25万基础地理数据（水系、交通、居民地等，需注册）；（3）资源环境科学与数据中心（resdc.cn）——中科院提供的中国土地利用、气象、土壤、DEM等数据；（4）Natural Earth（naturalearthdata.com）——全球1:10m、1:50m、1:110m文化（国界、城市）和自然（河流、湖泊）矢量数据，制图首选；（5）DIVA-GIS（diva-gis.org）——可按国家下载行政边界、道路、铁路、人口等数据；（6）GADM（gadm.org）——全球行政边界数据，从国家到村级多级别；（7）中国气象数据网（data.cma.cn）——气象站观测数据（需注册）；（8）NASA SEDAC——人口密度（GPWv4）、GDP等社会经济栅格数据。对于遥感影像，推荐USGS EarthExplorer（Landsat）和ESA Copernicus Hub（Sentinel）。注意检查数据的坐标系、精度和更新日期再使用。",
        "keywords": ["GIS数据下载", "免费数据源", "OSM", "Natural Earth", "GADM", "DIVA-GIS", "数据获取"]
    },

    # ===== 数据模型补充 =====
    {
        "id": "gis-124",
        "title": "数字高程模型（DEM）及其应用",
        "category": "data_types",
        "content": "DEM（Digital Elevation Model，数字高程模型）是表示地表高程的栅格数据，每个像素值代表该位置的海拔高度。常见的全球免费DEM数据源：（1）SRTM（Shuttle Radar Topography Mission）——30米分辨率，覆盖60°N-56°S，应用最广泛；（2）ASTER GDEM——30米分辨率，全球覆盖，但部分区域有云层遮挡噪声；（3）ALOS World 3D（AW3D30）——30米分辨率，日本JAXA发布，精度较高；（4）TanDEM-X——12米分辨率，部分区域达到更高精度，需申请；（5）中国区域的NASA DEM和COP-DEM（30米）。DEM的主要应用：坡度/坡向分析、流域划分、河网提取、地形阴影图（Hillshade）、视域分析、挖填方计算、洪水淹没模拟等。使用DEM前应检查是否有空洞（无数据区域），可用QGIS「栅格→分析→填充无数据」工具或GDAL的gdal_fillnodata.py修复。",
        "keywords": ["DEM", "高程模型", "SRTM", "ASTER", "坡度", "流域", "Hillshade"]
    },
    {
        "id": "gis-125",
        "title": "Shapefile和GeoPackage对比",
        "category": "data_types",
        "content": "Shapefile是ESRI于1990年代提出的矢量数据格式，至今仍是GIS中最常见的交换格式。它实际上包含至少三个文件（.shp几何、.dbf属性表、.shx索引），通常还有.prj投影信息、.cpg编码等辅助文件。Shapefile的局限性：单个文件最大2GB、字段名最长10个字符、不支持NULL值和时间戳类型、不支持拓扑关系。GeoPackage（.gpkg）是OGC制定的现代开放格式（2014年发布），基于SQLite数据库，优势包括：单个文件包含所有数据（无需多文件）、文件大小理论上无限、字段名无长度限制、支持完整SQL查询、可同时存储矢量和栅格数据、支持空间索引自动创建。建议：新项目使用GeoPackage作为主格式，仅在与旧系统交换时导出Shapefile。QGIS和ArcGIS Pro均完全支持GeoPackage的读写。转换方法：QGIS中右键图层→导出→另存为，格式选择GPKG。",
        "keywords": ["Shapefile", "GeoPackage", "GPKG", "矢量格式", "OGC标准", "数据交换"]
    },

    # ===== ArcGIS 补充 =====
    {
        "id": "gis-126",
        "title": "ArcGIS中空间校正和地理配准",
        "category": "arcgis_ops",
        "content": "在ArcGIS中将无坐标系的CAD数据或扫描地图赋予正确坐标的方法：（1）地理配准（Georeferencing）用于栅格（扫描地图/航片）：加载已知坐标的参考数据→「影像→地理配准」→添加控制点（选择图片上可识别的地物→对应到参考数据同一位置），通常需要4-6个均匀分布的控制点→选择变换方法（一阶多项式（仿射）最少3个点，适合大多数情况；样条函数适合需精确局部拟合；校正用橡皮页变换）→「更新地理配准」保存；（2）空间校正（Spatial Adjustment）用于矢量数据：编辑器打开目标图层→「编辑→空间校正」→设置校正方法（仿射变换（最少3对控制点）、相似变换（2对控制点）、橡皮页变换（多对控制点）。新建位移链接→选择调整方法→「校正」。两种操作的精度取决于控制点的数量和分布，可检查残差来判断配准质量（残差应控制在地图精度要求以内）。",
        "keywords": ["ArcGIS地理配准", "空间校正", "控制点", "仿射变换", "残差", "橡皮页变换"]
    },
    {
        "id": "gis-127",
        "title": "ArcGIS中创建缓冲区和服务区分析",
        "category": "arcgis_ops",
        "content": "ArcGIS中缓冲区和服务区分析的操作方法：（1）创建缓冲区——「分析→邻域分析→缓冲区」（ArcMap）或「分析→工具→缓冲区」（ArcGIS Pro）。支持：固定距离缓冲（如道路两侧50米）、基于属性字段的变距缓冲（如不同等级道路使用不同距离）、多环缓冲（一次创建多个同心缓冲环）、融合选项（相邻缓冲是否合并）；（2）服务区分析（Service Area）——属于网络分析扩展，基于路网计算等时圈/等距圈。「网络分析→新建服务区」→加载设施点→设置阻抗（时间/距离/其他成本）→设置间断值（如5分钟、10分钟、15分钟等时圈）→求解，即可生成从设施点出发沿路网的可达范围多边形；（3）欧氏距离和网络距离的区别：普通缓冲区是直线（欧氏）距离，服务区分析是沿实际路网的距离，两者结果可能有天壤之别。缓冲区分析不需要Network Analyst扩展，服务区分析需要。QGIS中ORSTools和QNEAT3插件提供类似功能。",
        "keywords": ["ArcGIS缓冲区", "服务区分析", "等时圈", "网络分析", "多环缓冲"]
    },

    # ===== QGIS 补充 =====
    {
        "id": "gis-128",
        "title": "QGIS中处理栅格数据的常用操作",
        "category": "qgis_ops",
        "content": "QGIS中栅格处理的常用操作：（1）裁剪栅格——「栅格→提取→按掩膜图层裁剪」，使用矢量面作为裁剪边界；或「按范围裁剪」使用矩形范围；（2）重采样——「栅格→投影→重投影」，可同时进行投影转换和重采样（最近邻、双线性、三次卷积），修改像元大小；（3）栅格计算器——「栅格→栅格计算器」，支持多波段运算和多栅格运算，如计算NDVI：('band5@1'-'band4@1')/('band5@1'+'band4@1')；（4）栅格转矢量——「栅格→转换→多边形化」，将分类栅格转为面矢量，适合将土地利用分类结果矢量化；（5）地形分析——「栅格→地形分析」下的坡度、坡向、山体阴影、可见度、地形起伏度等工具，几乎覆盖了ArcGIS Spatial Analyst的核心功能；（6）样式渲染——栅格属性→符号化，支持单波段灰度/伪彩色、多波段真彩色、山体阴影叠加等渲染方式。QGIS 3.x的栅格处理引擎基于GDAL，性能良好。大批量栅格处理建议直接使用GDAL命令行工具。",
        "keywords": ["QGIS栅格", "栅格计算器", "地形分析", "重采样", "按掩膜裁剪", "矢量转栅格"]
    },
    {
        "id": "gis-129",
        "title": "QGIS和Python结合自动化",
        "category": "qgis_ops",
        "content": "在QGIS中使用Python进行自动化处理主要有两种方式：（1）QGIS内置Python控制台——「插件→Python控制台」或Ctrl+Alt+P，可以在QGIS运行时操作当前项目的图层、属性和工具。示例：读取图层 layer = iface.activeLayer()，遍历要素 for f in layer.getFeatures(): print(f['字段名'])；（2）PyQGIS独立脚本——无需启动QGIS界面即可调用QGIS的处理工具。需要在脚本开头设置QGIS环境路径。示例批量投影转换脚本：import processing; processing.run('native:reprojectlayer', {'INPUT': input_path, 'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:4490'), 'OUTPUT': output_path})；（3）Processing Toolbox脚本——「处理→工具箱→创建新脚本」，可将常用工作流保存为可重复使用的工具，支持参数化输入。Python在GIS自动化中的典型场景：批量文件格式转换、批量投影转换、按模板生成地图、自动化数据质检（检查要素是否有几何错误、属性缺失等）。QGIS的Python API文档可在qgis.org/pyqgis查看。",
        "keywords": ["PyQGIS", "QGIS Python", "自动化", "processing", "Python控制台", "批量处理"]
    },

    # ===== GPS/GNSS 补充 =====
    {
        "id": "gis-130",
        "title": "差分GPS与RTK定位技术",
        "category": "gis_basics",
        "content": "差分GPS（DGPS）和RTK（Real-Time Kinematic）是提高GPS定位精度的关键技术：（1）普通GPS定位精度约3-10米，受电离层延迟、卫星轨道误差、多路径效应等因素影响；（2）DGPS原理——在地面已知精确坐标的基准站上观测GPS误差，通过无线电将修正信息广播给周围的移动站，移动站据此修正自身位置，典型精度达到亚米级（0.5-1米）；（3）RTK原理——利用载波相位观测值进行差分定位，基准站和移动站同时接收GPS信号，通过实时解算整周模糊度获得厘米级相对位置。RTK典型精度：水平1-2厘米，垂直2-3厘米。RTK需要基准站和移动站之间距离一般不超过10-20公里；（4）网络RTK（CORS）——利用多个连续运行参考站的网络解算区域误差模型，扩大服务范围。中国已建设超过5000个CORS站，大多数省份有省级CORS网络（如浙江ZJCORS、广东GDCORS），注册后可免费使用；（5）PPP（精密单点定位）——不需要本地基准站，利用精密星历和钟差产品实现厘米级定位，但收敛时间较长（15-30分钟），适合无CORS覆盖的偏远地区。",
        "keywords": ["DGPS", "RTK", "CORS", "差分定位", "厘米级定位", "载波相位", "PPP"]
    },
    {
        "id": "gis-131",
        "title": "GPS数据导入GIS的方法",
        "category": "gis_basics",
        "content": "将GPS采集的数据导入GIS进行处理的常见方法：（1）手持GPS导出——大部分手持GPS（Garmin等）支持导出GPX格式，QGIS中「矢量→GPS→GPS工具」或直接拖入GPX文件，可选择加载航点、航迹或航线图层；（2）手机APP数据——户外助手/两步路等APP支持导出KML/KMZ/GPX格式，直接拖入QGIS即可加载；奥维地图支持导出DXF/KML等格式；（3）RTK测量数据——通常导出为CSV格式（含点名、X/Y/Z坐标或经纬度），使用QGIS「图层→添加图层→添加分隔文本图层」导入，需正确指定X字段（经度/东坐标）和Y字段（纬度/北坐标）及坐标系；（4）坐标转换——野外采集通常是WGS84经纬度，导入后如需要投影坐标（如建筑设计的施工坐标系），使用「导出→另存为」在导出时选择目标CRS即可一次性完成投影转换；（5）精度检查——导入后建议叠加高分辨率底图或已有控制点数据检查点位精度，对精度不满足要求的点进行重测或剔除。    ",
        "keywords": ["GPS导入", "GPX", "KML", "RTK数据", "手持GPS", "坐标转换"]
    },

    # ===== 数据分析补充 =====
    {
        "id": "gis-132",
        "title": "空间插值方法对比",
        "category": "gis_basics",
        "content": "空间插值是用已知采样点的值估算未知位置的值，是GIS空间分析的核心方法之一。常用插值方法及适用场景：（1）反距离加权（IDW）——根据距离远近来加权，离采样点越近权重越大。优点是计算快速、结果直观；缺点是易受采样点分布不均匀影响，产生「牛眼」效应。适合采样点密集均匀的场景如污染物浓度插值；（2）克里金（Kriging）——基于地统计学的最优无偏估计方法，不仅考虑距离还考虑采样点的空间自相关结构（通过半变异函数建模）。优点是精度最高、提供预测误差估计（克里金方差）；缺点是需要较多的采样点（通常30个以上）且参数设置较复杂。适合高精度需求的地质矿产估算、土壤属性制图；（3）样条函数（Spline）——生成通过所有采样点的光滑曲面，适合表现连续变化的表面如地形高程；（4）自然邻域法（Natural Neighbor）——基于Voronoi图加权，适合分布不均匀的采样点；（5）趋势面分析——用多项式拟合全局趋势，适合提取大尺度空间格局。QGIS中所有插值工具在「处理→工具箱」搜索「插值」或使用SAGA/GRASS工具。ArcGIS中在「Spatial Analyst→插值」或「地统计向导」。",
        "keywords": ["空间插值", "IDW", "克里金", "Kriging", "样条函数", "反距离加权", "地统计"]
    },
    {
        "id": "gis-133",
        "title": "热点分析与空间自相关",
        "category": "gis_basics",
        "content": "热点分析和空间自相关是识别地理现象空间聚集模式的重要方法：（1）全局空间自相关（Global Moran's I）——衡量整个研究区内是否存在空间聚集。Moran's I值范围[-1, 1]，正值表示相似值聚集（高值靠近高值），负值表示离散（高值靠近低值），0表示随机分布。同时报告Z值和P值进行显著性检验；（2）局部空间自相关（LISA, Local Indicators of Spatial Association）——识别局部的聚集类型：高-高聚集（热点）、低-低聚集（冷点）、高-低异常（高高值被低值包围）、低-高异常。通常用LISA聚集图展示；（3）Getis-Ord Gi*热点分析——识别统计显著的「热点」（高值聚集）和「冷点」（低值聚集），生成每个要素的Z分数，Z>1.96（95%置信）或Z>2.58（99%置信）为显著热点；（4）分析工具——ArcGIS中有全套空间统计工具（Spatial Statistics工具箱）；QGIS需借助GeoDa软件（专为空间统计设计）或R语言的spdep包。应用场景：犯罪热点分析、疾病空间聚集检测、房价空间分异、商铺选址等。重要前提：分析前需选择合适的空间权重矩阵（距离阈值或邻接关系），不同权重矩阵可能得出不同结论。",
        "keywords": ["热点分析", "Moran's I", "LISA", "空间自相关", "Getis-Ord", "冷热点", "空间统计"]
    },
]

# ===== 合并、去重、写入 =====
all_docs = existing_docs + new_docs

# 去重（保留第一个出现的）
seen = set()
deduped = []
for doc in all_docs:
    if doc["id"] not in seen:
        seen.add(doc["id"])
        deduped.append(doc)

# 按 id 排序
deduped.sort(key=lambda x: x["id"])

os.makedirs(os.path.dirname(KB_PATH), exist_ok=True)

with open(KB_PATH, "w", encoding="utf-8") as f:
    json.dump(deduped, f, ensure_ascii=False, indent=2)

# 统计
cats = {}
for d in deduped:
    cats[d["category"]] = cats.get(d["category"], 0) + 1

print(f"\n{'='*50}")
print(f"知识库生成完成！")
print(f"总文档数: {len(deduped)}")
print(f"\n各类别统计:")
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count}")
print(f"{'='*50}")
