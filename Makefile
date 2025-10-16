SHELL := /bin/bash

# ===== Python & 环境 =====
PYTHON ?= $(shell which python)
PY_ABS := $(strip $(PYTHON))

# ===== 目录与文件 =====
OUT_DIR    ?= outputs
PIPE_DIR   ?= src/pipelines
LAS        ?= $(OUT_DIR)/synth_osm_corridor.las
LAZ_COPC   ?= $(OUT_DIR)/synth_osm_corridor.copc.laz
LAZ_ZIP    ?= $(OUT_DIR)/synth_osm_corridor_utm.laz
DTM        ?= $(OUT_DIR)/aoi_dtm.tif
DTM_FAST   ?= $(OUT_DIR)/aoi_dtm_fast.tif
SLOPE      ?= $(OUT_DIR)/aoi_slope.tif
HILLSHADE  ?= $(OUT_DIR)/aoi_hillshade.tif
HAG_LAS    ?= $(OUT_DIR)/synth_osm_corridor_hag.las
HAG_COPC   ?= $(OUT_DIR)/synth_osm_corridor_hag.copc.laz
HAG_LAZ    ?= $(OUT_DIR)/synth_osm_corridor_hag.laz

# 影像（请在 QGIS 导出当前 AOI 的正射为该路径）
RGB        ?= data/imagery/aoi_rgb.tif

# 采样输出（点属性特征）
NPZ        ?= $(OUT_DIR)/train_points.npz

# 输入矢量
AOI_GEO    ?= data/aoi/aoi_export_bbox.geojson
OSM_LINES  ?= data/osm/aoi_export_power_lines.geojson
OSM_TOWERS ?= data/osm/aoi_export_power_towers.geojson  # 可无

# 合成参数（单位：米）
WIDTH       ?= 50
GROUND_STEP ?= 4
VEG_DENS    ?= 0.15
SPAN        ?= 350
SAG         ?= 15
HEIGHT      ?= 28
PHASE_OFF   ?= 1.5

.PHONY: help env check pipelines synth_osm dem slope hillshade sample laz info hag info_hag hag_laz clean

help:
	@echo "make env           # 创建/更新conda环境（可选）"
	@echo "make check         # 检查输入数据存在"
	@echo "make synth_osm     # 合成走廊点云 -> $(LAS)"
	@echo "make pipelines     # 生成 PDAL JSON 管线（SMRF）"
	@echo "make dem           # SMRF 提地 -> DTM: $(DTM)"
	@echo "make slope         # DTM -> 坡度: $(SLOPE)"
	@echo "make hillshade     # DTM -> 阴影: $(HILLSHADE)"
	@echo "make hag           # 采样 DTM 计算 HAG -> $(HAG_LAS)"
	@echo "make hag_laz       # 压缩 HAG 点云（COPC优先）"
	@echo "make laz           # 压缩原始合成点云（COPC优先）"
	@echo "make info          # 快速检查输出"
	@echo "make info_hag      # 检查 HAG 维度"
	@echo "make clean         # 清理 outputs"

env:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Activate: conda activate ot_corridor_min"

check:
	@test -f "$(AOI_GEO)"   || (echo "Missing $(AOI_GEO)"; exit 1)
	@test -f "$(OSM_LINES)" || (echo "Missing $(OSM_LINES)"; exit 1)
	@echo "[OK] inputs present."

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

$(PIPE_DIR):
	mkdir -p "$(PIPE_DIR)"

# 自动生成 PDAL JSON（稳妥的 printf 写入；在 SMRF 前补 assign 消除回波告警）
pipelines: $(PIPE_DIR)
	@mkdir -p "$(PIPE_DIR)"
	@echo "Write $(PIPE_DIR)/smrf_dtm.json (no assign)"
	@printf '%s\n' \
	  '{ "pipeline": [' \
	  '  { "type":"readers.las" },' \
	  '  { "type":"filters.smrf", "scalar":1.25, "slope":0.15, "threshold":0.5, "window":16 },' \
	  '  { "type":"filters.range", "limits":"Classification[2:2]" },' \
	  '  { "type":"writers.gdal",' \
	  '    "gdaldriver":"GTiff",' \
	  '    "output_type":"min",' \
	  '    "resolution":1.0,' \
	  '    "nodata":-9999,' \
	  '    "dimension":"Z",' \
	  '    "data_type":"float32"' \
	  '  }' \
	  ']}' > "$(PIPE_DIR)/smrf_dtm.json"


# ===== 合成走廊点云 =====
synth_osm: check $(OUT_DIR)
	@bash -lc '\
	  set -euo pipefail; \
	  PY="$(PY_ABS)"; \
	  echo "==[Synthesizing corridor]=="; \
	  echo "PYTHON: $$PY"; \
	  if [ ! -x "$$PY" ]; then echo "ERROR: python 不存在: $$PY"; exit 1; fi; \
	  echo "AOI    : $(AOI_GEO)"; \
	  echo "LINES  : $(OSM_LINES)"; \
	  echo "TOWERS : $(OSM_TOWERS)"; \
	  echo "OUT_LAS: $(LAS)"; \
	  echo "----------------------------------------------"; \
	  mkdir -p "$(OUT_DIR)"; \
	  $$PY src/synth_from_osm.py \
	    --aoi "$(AOI_GEO)" \
	    --lines "$(OSM_LINES)" \
	    --towers "$(OSM_TOWERS)" \
	    --out "$(LAS)" \
	    --width $(WIDTH) --ground-step $(GROUND_STEP) \
	    --veg-density $(VEG_DENS) --span $(SPAN) --sag $(SAG) \
	    --height $(HEIGHT) --phase-offset $(PHASE_OFF); \
	  echo "----------------------------------------------"; \
	  if [ ! -s "$(LAS)" ]; then echo "ERROR: 未生成或为空：$(LAS)"; exit 1; fi; \
	  echo "OK: 生成成功 -> $(LAS)"; \
	'

# ===== DTM（SMRF）=====
dem: synth_osm pipelines
	@if command -v pdal >/dev/null 2>&1; then \
	  echo "-> DTM via PDAL SMRF: $(DTM)"; \
	  pdal pipeline "$(PIPE_DIR)/smrf_dtm.json" \
	    --readers.las.filename="$(LAS)" \
	    --writers.gdal.filename="$(DTM)"; \
	else \
	  echo "PDAL not found; cannot build DTM."; exit 1; \
	fi

dem_fast: synth_osm
	@if command -v pdal >/dev/null 2>&1; then \
	  echo "-> DTM from class=2 (no SMRF): $(DTM_FAST)"; \
	  tmp=$$(mktemp /tmp/dem_fast.XXXXXX.json); \
	  printf '%s\n' \
	    '{ "pipeline": [' \
	    '  { "type":"readers.las", "filename":"$(LAS)" },' \
	    '  { "type":"filters.range", "limits":"Classification[2:2]" },' \
	    '  { "type":"writers.gdal",' \
	    '    "filename":"$(DTM_FAST)",' \
	    '    "gdaldriver":"GTiff",' \
	    '    "output_type":"min",' \
	    '    "resolution":1.0,' \
	    '    "nodata":-9999,' \
	    '    "dimension":"Z",' \
	    '    "data_type":"float32"' \
	    '  }' \
	    ']}' > "$$tmp"; \
	  pdal pipeline "$$tmp"; \
	  rm -f "$$tmp"; \
	else echo "PDAL not found"; exit 1; fi



# ===== 坡度（GDAL）=====
slope: dem
	@if command -v gdaldem >/dev/null 2>&1; then \
	  echo "-> Slope from DTM: $(SLOPE)"; \
	  gdaldem slope "$(DTM)" "$(SLOPE)" -s 1.0 -of GTiff -compute_edges; \
	else \
	  echo "gdaldem not found; install GDAL (conda-forge)."; exit 1; \
	fi

# ===== 阴影（GDAL，可选）=====
hillshade: dem
	@if command -v gdaldem >/dev/null 2>&1; then \
	  echo "-> Hillshade from DTM: $(HILLSHADE)"; \
	  gdaldem hillshade "$(DTM)" "$(HILLSHADE)" -z 1.0 -s 1.0 -of GTiff -compute_edges; \
	else \
	  echo "gdaldem not found; skip hillshade."; \
	fi

# ===== 采样影像/坡度到点（可选，需你提供脚本）=====
sample: slope
	@if [ ! -f "src/sample_raster_to_points.py" ]; then \
	  echo "缺少 src/sample_raster_to_points.py；请放入采样脚本或跳过该目标。"; exit 1; \
	fi
	@if [ ! -f "$(RGB)" ]; then \
	  echo "缺少影像 $(RGB)；请在 QGIS 导出 AOI 正射至该路径。"; exit 1; \
	fi
	"$(PY_ABS)" src/sample_raster_to_points.py \
	  --las "$(LAS)" --dtm "$(DTM)" --slope "$(SLOPE)" --rgb "$(RGB)" --out "$(NPZ)"

# ===== 压缩点云（COPC 优先）=====
laz: synth_osm
	@bash -lc '\
	  set -e; \
	  echo "Try COPC (.copc.laz)"; \
	  if pdal translate "$(LAS)" "$(LAZ_COPC)"; then \
	    echo "OK -> $(LAZ_COPC)"; \
	  else \
	    echo "COPC failed; try LAZ with writers.las"; \
	    pdal translate "$(LAS)" "$(LAZ_ZIP)" --writers.las.compression=true; \
	    echo "OK -> $(LAZ_ZIP)"; \
	  fi \
	'

# ===== HAG：用 rasterio 采样 DTM，写 ExtraBytes =====
hag: dem
	@if [ ! -f "src/add_hag.py" ]; then \
	  echo "缺少 src/add_hag.py（HAG 计算脚本）。"; \
	  echo "请将我之前提供的 add_hag.py 保存到 src/ 目录后再运行 make hag。"; \
	  exit 1; \
	fi
	@bash -lc '\
	  set -euo pipefail; \
	  echo "-> Add HAG using rasterio from $(DTM)"; \
	  "$(PY_ABS)" src/add_hag.py \
	    --in "$(LAS)" --dtm "$(DTM)" --out "$(HAG_LAS)"; \
	  test -s "$(HAG_LAS)" && echo "OK -> $(HAG_LAS)"; \
	'

# ===== HAG 压缩（COPC 优先）=====
hag_laz: hag
	@bash -lc '\
	  set -e; \
	  echo "Try COPC (.copc.laz)"; \
	  if pdal translate "$(HAG_LAS)" "$(HAG_COPC)"; then \
	    echo "OK -> $(HAG_COPC)"; \
	  else \
	    echo "COPC failed; try LAZ with writers.las"; \
	    pdal translate "$(HAG_LAS)" "$(HAG_LAZ)" --writers.las.compression=true; \
	    echo "OK -> $(HAG_LAZ)"; \
	  fi \
	'

# ===== 快速检查 =====
info:
	@if command -v pdal >/dev/null 2>&1; then \
	  echo "---- LAS ----"; \
	  test -f "$(LAS)" && pdal info "$(LAS)" --summary | egrep -i "count|minx|maxx|miny|maxy" || true; \
	  echo "---- COPC ----"; \
	  test -f "$(LAZ_COPC)" && pdal info "$(LAZ_COPC)" --summary | egrep -i "count|minx|maxx|miny|maxy" || true; \
	  echo "---- LAZ ----"; \
	  test -f "$(LAZ_ZIP)" && pdal info "$(LAZ_ZIP)" --summary | egrep -i "count|minx|maxx|miny|maxy" || true; \
	else \
	  echo "PDAL not installed; skip pdal info."; \
	fi
	@if command -v gdalinfo >/dev/null 2>&1; then \
	  echo "---- DTM ----"; test -f "$(DTM)" && gdalinfo -stats "$(DTM)" | egrep "Size|Corner|Mean|StdDev|Minimum|Maximum" || true; \
	  echo "---- SLOPE ----"; test -f "$(SLOPE)" && gdalinfo -stats "$(SLOPE)" | egrep "Size|Corner|Mean|StdDev|Minimum|Maximum" || true; \
	fi

info_hag:
	@if command -v pdal >/dev/null 2>&1; then \
	  test -f "$(HAG_LAS)" && pdal info "$(HAG_LAS)" --dimensions | egrep -i "HeightAboveGround|Classification|Z" || true; \
	else \
	  echo "PDAL not installed; skip pdal info."; \
	fi

clean:
	rm -f "$(OUT_DIR)"/synth_osm_corridor*.las "$(OUT_DIR)"/synth_osm_corridor*.laz
	rm -f "$(DTM)" "$(SLOPE)" "$(HILLSHADE)"
