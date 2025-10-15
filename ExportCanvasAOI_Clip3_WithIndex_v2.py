# -*- coding: utf-8 -*-
"""
QGIS Processing tool:
以“当前画布范围”为 AOI，裁剪并存储植被/电力线路/电力塔三类数据，并输出 AOI BBOX。
适用：QGIS 3.22+（含 3.40.x），macOS/Windows/Linux。

输出（示例，前缀= aoi_export，输出坐标系= EPSG:4326）：
  <out_dir>/aoi_export_bbox.geojson
  <out_dir>/aoi_export_vegetation.geojson  （若提供植被图层）
  <out_dir>/aoi_export_power_lines.geojson  （若提供线路图层）
  <out_dir>/aoi_export_power_towers.geojson （若提供塔图层）
"""

# -*- coding: utf-8 -*-
"""
QGIS Processing tool:
以当前画布范围为 AOI，对植被(面)/线路(线)/塔(点)三类图层进行裁剪并保存；同时输出 AOI BBOX。
适用：QGIS 3.22+（含 3.40.x），macOS/Windows/Linux。
"""
# -*- coding: utf-8 -*-
"""
QGIS Processing tool:
以当前画布范围为 AOI，对植被(面)/线路(线)/塔(点)三类图层进行裁剪并保存；自动尝试为输入图层创建空间索引；
同时输出 AOI BBOX（GeoJSON）。
适用：QGIS 3.22+（含 3.40.x），macOS/Windows/Linux。
"""
# -*- coding: utf-8 -*-
"""
QGIS Processing tool:
以当前画布范围为 AOI，裁剪并保存植被(面)/线路(线)/塔(点)，可选：导出后自动执行 make 合成点云并加载 LAZ。
适用：QGIS 3.22+（含 3.40.x），macOS/Windows/Linux。
"""

# -*- coding: utf-8 -*-
"""
QGIS Processing tool:
以当前画布范围为 AOI，对植被(面)/线路(线)/塔(点)三类图层进行裁剪并保存；自动尝试为输入图层创建空间索引；
同时输出 AOI BBOX（GeoJSON）。
适用：QGIS 3.22+（含 3.40.x），macOS/Windows/Linux。
"""

# -*- coding: utf-8 -*-
"""
QGIS Processing tool:
以当前画布范围为 AOI，裁剪并保存植被(面)/线路(线)/塔(点)，可选：导出后自动执行 make 合成点云并加载 LAZ。
适用：QGIS 3.22+（含 3.40.x），macOS/Windows/Linux。
"""

import shlex, subprocess
from pathlib import Path

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProject,
    QgsRectangle,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterCrs,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingException,
    QgsFeature,
    QgsVectorLayer,
    QgsGeometry,
    QgsPointXY,
    QgsPointCloudLayer,
)
import processing


class ExportCanvasAOIClip3WithMake(QgsProcessingAlgorithm):
    # 输入/输出参数键
    P_VEG = "VEG_LAYER"
    P_LINE = "LINE_LAYER"
    P_TOWER = "TOWER_LAYER"
    P_OUTDIR = "OUT_DIR"
    P_OUTCRS = "OUT_CRS"
    P_PREFIX = "PREFIX"
    P_BUILD_INDEX = "BUILD_INDEX"

    # 自动合成相关
    P_RUN_MAKE = "RUN_MAKE"
    P_PROJECT_ROOT = "PROJECT_ROOT"
    P_MAKE_CMD = "MAKE_CMD"
    P_MAKE_TARGET = "MAKE_TARGET"
    P_LAZ_PATH = "LAZ_PATH"
    P_USE_SHELL = "USE_SHELL"

    def tr(self, text):
        return QCoreApplication.translate("ExportCanvasAOIClip3WithMake", text)

    def createInstance(self):
        return ExportCanvasAOIClip3WithMake()

    def name(self):
        return "export_canvas_aoi_clip_3layers_with_make"

    def displayName(self):
        return self.tr("画布AOI裁剪与存储（植被/线路/塔 · 可自动make合成点云）")

    def group(self):
        return self.tr("电力遥感 · 一键工具")

    def groupId(self):
        return "power_rs_suite"

    def shortHelpString(self):
        return self.tr(
            "以当前地图画布范围为 AOI，裁剪并保存植被/线路/塔（自动建索引可选）；"
            "可选：导出完成后自动执行 make synth_osm（或自定义目标），并加载生成的 LAZ 到工程。"
        )

    def initAlgorithm(self, config=None):
        # 三类输入图层（皆可选；为空则跳过）
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.P_VEG, self.tr("植被图层（面，optional）"),
                [QgsProcessing.TypeVectorPolygon], optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.P_LINE, self.tr("电力线路图层（线，optional）"),
                [QgsProcessing.TypeVectorLine], optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.P_TOWER, self.tr("电力塔图层（点，optional）"),
                [QgsProcessing.TypeVectorPoint], optional=True
            )
        )
        # 输出目录 + CRS + 前缀
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.P_OUTDIR, self.tr("输出目录（写入 GeoJSON）")
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.P_OUTCRS, self.tr("输出坐标系"),
                defaultValue=QgsCoordinateReferenceSystem("EPSG:4326")
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.P_PREFIX, self.tr("文件名前缀"), defaultValue="aoi_export"
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_BUILD_INDEX, self.tr("裁剪前为输入图层创建空间索引（支持的数据源）"),
                defaultValue=True
            )
        )

        # -------- 自动合成（make）相关参数 --------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_RUN_MAKE, self.tr("导出后自动执行 make 合成并加载 LAZ"),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.P_PROJECT_ROOT, self.tr("项目根目录（Makefile 所在目录）"), optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.P_MAKE_CMD, self.tr("make 命令（可填 conda run -n ot_osm_corridor make）"),
                defaultValue="make", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.P_MAKE_TARGET, self.tr("make 目标名"),
                defaultValue="synth_osm", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.P_LAZ_PATH, self.tr("LAZ 输出路径（相对/绝对；留空用默认 data/crops/synth_osm_corridor_4326.laz）"),
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_USE_SHELL, self.tr("通过 Shell 执行命令（勾选以支持 conda run 等复合命令）"),
                defaultValue=True
            )
        )

    # ---------- 获取画布范围（优先 iface，回退项目联合范围） ----------
    def _canvas_extent_in_crs(self, target_crs: QgsCoordinateReferenceSystem) -> QgsRectangle:
        src_crs = None
        rect = None
        # 1) GUI 画布
        try:
            from qgis.utils import iface
            if iface is not None and iface.mapCanvas() is not None:
                canvas = iface.mapCanvas()
                rect = QgsRectangle(canvas.extent())
                src_crs = canvas.mapSettings().destinationCrs()
        except Exception:
            rect = None
        # 2) 回退：项目所有图层的联合范围
        if rect is None or not rect.isFinite():
            proj = QgsProject.instance()
            if src_crs is None or not src_crs.isValid():
                src_crs = proj.crs()
            rect_all = None
            for lyr in proj.mapLayers().values():
                try:
                    e = lyr.extent()
                    if e and e.isFinite():
                        rect_all = QgsRectangle(e) if rect_all is None else rect_all.combineExtentWith(e) or rect_all
                except Exception:
                    pass
            if rect_all is None or not rect_all.isFinite():
                raise QgsProcessingException("无法获得画布或项目范围，请在 QGIS GUI 中运行或手动提供范围。")
            rect = rect_all
        # 3) 投影到目标 CRS
        if target_crs and target_crs.isValid() and src_crs and src_crs.isValid() and (src_crs != target_crs):
            tr = QgsCoordinateTransform(src_crs, target_crs, QgsProject.instance())
            rect = tr.transformBoundingBox(rect)
        return rect

    # ---------- 将矩形转为 extent 字串 ----------
    def _extent_str(self, rect: QgsRectangle, crs: QgsCoordinateReferenceSystem) -> str:
        # 规范格式：xmin,xmax,ymin,ymax [EPSG:xxxx]
        return f"{rect.xMinimum()},{rect.xMaximum()},{rect.yMinimum()},{rect.yMaximum()} [{crs.authid()}]"

    # ---------- 用矩形和 CRS 生成一个 AOI 面的内存图层 ----------
    def _rect_to_mem_layer(self, rect: QgsRectangle, crs: QgsCoordinateReferenceSystem, name="AOI"):
        vl = QgsVectorLayer(f"Polygon?crs={crs.authid()}", name, "memory")
        pr = vl.dataProvider()
        ring = [
            QgsPointXY(rect.xMinimum(), rect.yMinimum()),
            QgsPointXY(rect.xMinimum(), rect.yMaximum()),
            QgsPointXY(rect.xMaximum(), rect.yMaximum()),
            QgsPointXY(rect.xMaximum(), rect.yMinimum()),
            QgsPointXY(rect.xMinimum(), rect.yMinimum()),
        ]
        geom = QgsGeometry.fromPolygonXY([ring])
        f = QgsFeature(); f.setGeometry(geom)
        pr.addFeatures([f]); vl.updateExtents()
        return vl

    # ---------- 尝试为输入图层创建空间索引 ----------
    def _ensure_spatial_index(self, layer, context: QgsProcessingContext, feedback: QgsProcessingFeedback):
        try:
            processing.run("native:createspatialindex", {"INPUT": layer}, context=context, feedback=feedback)
            feedback.pushInfo(f"已为图层创建空间索引：{layer.name()}")
        except Exception:
            feedback.pushInfo(f"跳过空间索引：{layer.name()}（数据源可能不支持）")

    # ---------- 按矩形裁剪并保存 ----------
    def _clip_and_save(self, layer, rect_in_layer_crs: QgsRectangle,
                       out_crs: QgsCoordinateReferenceSystem, out_path: Path,
                       context: QgsProcessingContext, feedback: QgsProcessingFeedback):
        extent_param = self._extent_str(rect_in_layer_crs, layer.crs())
        # 1) 先按范围提取并裁剪
        res = processing.run(
            "native:extractbyextent",
            {"INPUT": layer, "EXTENT": extent_param, "CLIP": True, "OUTPUT": "TEMPORARY_OUTPUT"},
            context=context, feedback=feedback,
        )
        clipped = res["OUTPUT"]
        # 2) 重投影到目标 CRS 并写文件
        processing.run(
            "native:reprojectlayer",
            {"INPUT": clipped, "TARGET_CRS": out_crs, "OUTPUT": str(out_path)},
            context=context, feedback=feedback,
        )

    # ---------- 执行 make 并尝试加载 LAZ ----------
    def _run_make_and_load(self, project_root: Path, make_cmd: str, make_target: str,
                           aoi_path: Path, lines_path: Path, towers_path: Path or None,
                           laz_path: Path or None, use_shell: bool, feedback: QgsProcessingFeedback):
        # 构造命令：make -C <root> -B <target> AOI_GEO=... OSM_LINES=... [OSM_TOWERS=...]
        cmd = f'{make_cmd} -C {shlex.quote(str(project_root))} -B {shlex.quote(make_target)} ' \
              f'AOI_GEO={shlex.quote(str(aoi_path))} OSM_LINES={shlex.quote(str(lines_path))}'
        if towers_path and Path(towers_path).exists():
            cmd += f' OSM_TOWERS={shlex.quote(str(towers_path))}'
        feedback.pushInfo("执行命令：" + cmd)

        # 执行
        if use_shell:
            proc = subprocess.run(cmd, shell=True, cwd=str(project_root),
                                  capture_output=True, text=True)
        else:
            proc = subprocess.run(shlex.split(cmd), shell=False, cwd=str(project_root),
                                  capture_output=True, text=True)
        feedback.pushInfo("STDOUT:\n" + (proc.stdout or "(empty)"))
        if proc.stderr:
            feedback.pushWarning("STDERR:\n" + proc.stderr)
        if proc.returncode != 0:
            raise QgsProcessingException(f"make 返回非零退出码：{proc.returncode}")

        # LAZ 路径：优先参数，若未给则使用项目默认 data/crops/synth_osm_corridor_4326.laz
        if laz_path is None or str(laz_path).strip() == "":
            laz_path = project_root / "data" / "crops" / "synth_osm_corridor_4326.laz"
        laz_path = Path(laz_path)

        if not laz_path.exists() or laz_path.stat().st_size < 500:
            feedback.reportError(f"未找到有效 LAZ：{laz_path}（可能为空文件或路径不一致）")
            return str(laz_path)

        # 自动加载到工程
        try:
            layer_name = laz_path.stem
            pcl = QgsPointCloudLayer(str(laz_path), layer_name, "pdal")
            if pcl.isValid():
                QgsProject.instance().addMapLayer(pcl)
                feedback.pushInfo(f"已加载点云：{laz_path}")
            else:
                feedback.pushWarning("点云加载失败，但文件已生成。请手动添加该 LAZ。")
        except Exception as _:
            feedback.pushWarning("当前 QGIS 版本未能通过脚本直接加载点云，请手动添加该 LAZ。")

        return str(laz_path)

    def processAlgorithm(self, params, context: QgsProcessingContext, feedback: QgsProcessingFeedback):
        # 读取参数
        veg_layer = self.parameterAsVectorLayer(params, self.P_VEG, context)
        line_layer = self.parameterAsVectorLayer(params, self.P_LINE, context)
        tower_layer = self.parameterAsVectorLayer(params, self.P_TOWER, context)
        out_dir = Path(self.parameterAsFileOutput(params, self.P_OUTDIR, context) or "")
        out_crs = self.parameterAsCrs(params, self.P_OUTCRS, context)
        prefix = self.parameterAsString(params, self.P_PREFIX, context) or "aoi_export"
        build_index = self.parameterAsBool(params, self.P_BUILD_INDEX, context)

        run_make = self.parameterAsBool(params, self.P_RUN_MAKE, context)
        project_root = self.parameterAsFileOutput(params, self.P_PROJECT_ROOT, context)
        make_cmd = self.parameterAsString(params, self.P_MAKE_CMD, context) or "make"
        make_target = self.parameterAsString(params, self.P_MAKE_TARGET, context) or "synth_osm"
        laz_path_str = self.parameterAsString(params, self.P_LAZ_PATH, context) or ""
        use_shell = self.parameterAsBool(params, self.P_USE_SHELL, context)

        if not out_dir:
            raise QgsProcessingException(self.tr("请指定输出目录。"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # 工程 CRS（若未设置就用输出 CRS）
        proj_crs = QgsProject.instance().crs()
        if not proj_crs.isValid():
            proj_crs = out_crs

        # 0) 计算 AOI（画布范围）在工程 CRS 下的矩形，并导出 AOI BBOX
        rect_proj = self._canvas_extent_in_crs(proj_crs)
        feedback.pushInfo(self.tr("导出 AOI（画布范围）…"))
        aoi_mem = self._rect_to_mem_layer(rect_proj, proj_crs, name="AOI_bbox")
        aoi_out = out_dir / f"{prefix}_bbox.geojson"
        processing.run(
            "native:reprojectlayer",
            {"INPUT": aoi_mem, "TARGET_CRS": out_crs, "OUTPUT": str(aoi_out)},
            context=context, feedback=feedback,
        )

        # 1) 植被（若提供）
        out_veg = None
        if veg_layer is not None:
            feedback.pushInfo(self.tr("裁剪植被…"))
            if build_index: self._ensure_spatial_index(veg_layer, context, feedback)
            rect_in_layer = self._canvas_extent_in_crs(veg_layer.crs())
            out_veg = out_dir / f"{prefix}_vegetation.geojson"
            self._clip_and_save(veg_layer, rect_in_layer, out_crs, out_veg, context, feedback)

        # 2) 线路（若提供）
        out_line = None
        if line_layer is not None:
            feedback.pushInfo(self.tr("裁剪电力线路…"))
            if build_index: self._ensure_spatial_index(line_layer, context, feedback)
            rect_in_layer = self._canvas_extent_in_crs(line_layer.crs())
            out_line = out_dir / f"{prefix}_power_lines.geojson"
            self._clip_and_save(line_layer, rect_in_layer, out_crs, out_line, context, feedback)

        # 3) 塔（若提供）
        out_tower = None
        if tower_layer is not None:
            feedback.pushInfo(self.tr("裁剪电力塔…"))
            if build_index: self._ensure_spatial_index(tower_layer, context, feedback)
            rect_in_layer = self._canvas_extent_in_crs(tower_layer.crs())
            out_tower = out_dir / f"{prefix}_power_towers.geojson"
            self._clip_and_save(tower_layer, rect_in_layer, out_crs, out_tower, context, feedback)

        results = {
            "AOI_BBOX": str(aoi_out),
            "VEGETATION": str(out_veg) if out_veg else "",
            "POWER_LINES": str(out_line) if out_line else "",
            "POWER_TOWERS": str(out_tower) if out_tower else "",
            "LAZ": "",
        }

        # 4) （可选）执行 make 并加载 LAZ
        if run_make:
            if not project_root:
                raise QgsProcessingException("已勾选“自动执行 make”，但未提供“项目根目录（Makefile 所在目录）”。")
            project_root = Path(project_root)

            if not out_line or not Path(out_line).exists():
                raise QgsProcessingException("自动执行 make 需要线路输出（*_power_lines.geojson）。请确保已提供线路图层并成功导出。")

            # 组装参数并执行
            laz_path = Path(laz_path_str) if laz_path_str else None
            laz_path_str = self._run_make_and_load(
                project_root=project_root,
                make_cmd=make_cmd,
                make_target=make_target,
                aoi_path=aoi_out,
                lines_path=Path(out_line),
                towers_path=Path(out_tower) if out_tower else None,
                laz_path=laz_path,
                use_shell=use_shell,
                feedback=feedback,
            )
            results["LAZ"] = laz_path_str

        # 汇总返回
        for k, v in results.items():
            if v:
                feedback.pushInfo(f"{k}: {v}")
        return results
