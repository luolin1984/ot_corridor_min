SHELL := /bin/bash

# === 可按需覆盖 ===
PYTHON ?= $(shell which python)     # 如在 QGIS 里调用可用：PYTHON=/Users/.../envs/ot_corridor_min/bin/python
OUT_DIR ?= outputs
OUT_LAS ?= $(OUT_DIR)/synth_osm_corridor.las
OUT_LAZ ?= $(OUT_DIR)/synth_osm_corridor_utm.laz  # 若启用压缩

# 输入数据
AOI_GEO     ?= data/aoi/aoi_export_bbox.geojson
OSM_LINES   ?= data/osm/aoi_export_power_lines.geojson
OSM_TOWERS  ?= data/osm/aoi_export_power_towers.geojson  # 可不存在，脚本会自动跳过

# 合成参数（单位：米；输出坐标系自动 UTM，可改为指定 EPSG）
WIDTH       ?= 50       # 走廊半宽
GROUND_STEP ?= 4        # 地面点网格步长
VEG_DENS    ?= 0.15     # 植被点密度（点/平米）
SPAN        ?= 350      # 理想塔间距（用于导线采样的近似）
SAG         ?= 15       # 导线弧垂幅度
HEIGHT      ?= 28       # 导线/塔相对地面典型高度
PHASE_OFF   ?= 1.5      # 多相横向偏移

.PHONY: help env check synth_osm laz info clean

help:
	@echo "make env         # 创建/更新conda环境"
	@echo "make check       # 检查输入文件存在"
	@echo "make synth_osm   # 合成 .las（自动UTM）"
	@echo "make laz         # 若安装PDAL，则把 .las 压缩为 .laz"
	@echo "make info        # pdal info 查看统计（需安装PDAL）"
	@echo "make clean       # 清理 outputs"

env:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Activate: conda activate ot_corridor_min"

check:
	@test -f "$(AOI_GEO)"     || (echo "Missing $(AOI_GEO)"; exit 1)
	@test -f "$(OSM_LINES)"   || (echo "Missing $(OSM_LINES)"; exit 1)
	@echo "[OK] inputs present."

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

synth_osm: check $(OUT_DIR)
	@bash -lc '\
	  set -euo pipefail; \
	  echo "==[Synthesizing corridor]=="; \
	  echo "PYTHON: $(PYTHON)"; \
	  if [ -x "$(PYTHON)" ]; then "$(PYTHON)" -V || true; fi; \
	  echo "AOI    : $(AOI_GEO)"; \
	  echo "LINES  : $(OSM_LINES)"; \
	  echo "TOWERS : $(OSM_TOWERS)"; \
	  echo "OUT_LAS: $(OUT_LAS)"; \
	  echo "----------------------------------------------"; \
	  mkdir -p "$(OUT_DIR)"; \
	  "$(PYTHON)" src/synth_from_osm.py \
	    --aoi "$(AOI_GEO)" \
	    --lines "$(OSM_LINES)" \
	    --towers "$(OSM_TOWERS)" \
	    --out "$(OUT_LAS)" \
	    --width $(WIDTH) --ground-step $(GROUND_STEP) \
	    --veg-density $(VEG_DENS) --span $(SPAN) --sag $(SAG) \
	    --height $(HEIGHT) --phase-offset $(PHASE_OFF); \
	  echo "----------------------------------------------"; \
	  if [ ! -s "$(OUT_LAS)" ]; then \
	    echo "ERROR: 未生成输出或文件大小为0：$(OUT_LAS)"; \
	    exit 1; \
	  fi; \
	  echo "OK: 生成成功 -> $(OUT_LAS)"; \
	'


laz: synth_osm
	@if command -v pdal >/dev/null 2>&1; then \
	  echo "Compressing to LAZ -> $(OUT_LAZ)"; \
	  pdal translate "$(OUT_LAS)" "$(OUT_LAZ)"; \
	else \
	  echo "PDAL not found, skip LAZ compression."; \
	fi

info:
	@if command -v pdal >/dev/null 2>&1; then \
	  pdal info "$(OUT_LAS)" --summary || true; \
	  if [ -f "$(OUT_LAZ)" ]; then pdal info "$(OUT_LAZ)" --summary || true; fi \
	else \
	  echo "PDAL not installed."; \
	fi

clean:
	rm -f "$(OUT_DIR)"/synth_osm_corridor*
