# Makefile for HMAGAT Docker workflow

COMPOSE_FILE := docker/docker-compose.yml
DOCKER_COMPOSE := docker compose -f $(COMPOSE_FILE)
SERVICE := hmagat
CONTAINER_NAME ?= hmagat-work
MODEL ?= hmagat
SVG_OUTPUT_DIR ?= outputs/svg_$(MODEL)
SVG_INPUT ?= /workspace/$(SVG_OUTPUT_DIR)/anim_0.svg
GIF_OUTPUT ?= /workspace/outputs/anim_$(MODEL).gif
GIF_FRAMES_DIR ?= /workspace/outputs/gif_frames_$(MODEL)
GIF_FPS ?= 6
COMPARE_GIF_OUTPUT ?= /workspace/outputs/anim_compare.gif
COMPARE_GIF_SMALL_OUTPUT ?= /workspace/outputs/anim_compare_small.gif
COMPARE_GIF_MAX_WIDTH ?= 1024
COMPARE_GIF_SMALL_MAX_WIDTH ?= 768

# GPU selection: default to the strong GPU only
NVIDIA_VISIBLE_DEVICES ?= 0
CUDA_VISIBLE_DEVICES ?= 0

# Common demo args
COMMON_DEMO_ARGS = \
	--obs_radius 5 \
	--save_termination_state \
	--add_data_cost_to_go \
	--normalize_cost_to_go \
	--clamp_cost_to_go 1.0 \
	--use_lists \
	--device 0 \
	--run_online_expert \
	--model_residuals all \
	--use_edge_attr \
	--use_edge_attr_for_messages positions+manhattan \
	--edge_attr_cnn_mode MLP \
	--load_positions_separately \
	--train_on_terminated_agents \
	--recursive_oe \
	--cnn_mode ResNetLarge_withMLP \
	--collision_shielding pibt \
	--action_sampling probabilistic \
	--test_name one_demo \
	--test_num_samples 1 \
	--test_obs_radius 5 \
	--test_map_types warehouse=1.0 \
	--test_num_agents 32+32 \
	--test_wall_width_min 8 \
	--test_wall_width_max 8 \
	--test_vertical_gap 1 \
	--test_num_wall_rows_min 5 \
	--test_num_wall_rows_max 5 \
	--test_num_wall_cols_min 2 \
	--test_num_wall_cols_max 2 \
	--test_side_pad 3 \
	--test_max_episode_steps 256 \
	--test_min_dist 10

ifeq ($(MODEL),magat)
MODEL_LABEL = MAGAT
MODEL_DEMO_ARGS = \
	--checkpoints_dir checkpoints/magat \
	--run_name magat \
	--imitation_learning_model MAGAT
else
MODEL_LABEL = HMAGAT
MODEL_DEMO_ARGS = \
	--hypergraph_comm_radius 7 \
	--hyperedge_generation_method kmeans \
	--hypergraph_num_updates 10 \
	--hypergraph_wait_one \
	--hypergraph_initial_colperc 0.1 \
	--hypergraph_final_colperc 0.1 \
	--checkpoints_dir checkpoints/hmagat \
	--run_name hmagat \
	--imitation_learning_model DirectionalHMAGAT \
	--hyperedge_feature_generator magat \
	--final_feature_generator magat \
	--rl_based_temperature_sampling \
	--temperature_checkpoints_dir checkpoints/hmagat_temperature_module \
	--temperature_run_name simple_rl \
	--temperature_actor_critic simple-local-val-init \
	--temperature_optimize only-all-on-goal \
	--iterations_per_epoch 3 \
	--temperature_min_val 0.5 \
	--temperature_max_val 0.9 \
	--temperature_sampling_model_epoch_num 43
endif

DEMO_ARGS = $(COMMON_DEMO_ARGS) $(MODEL_DEMO_ARGS)

.PHONY: help build rebuild shell ps logs gpu-check demo demo-small demo-svg gif-tools gif gif-compare gif-compare-small

help:
	@echo "🐳 HMAGAT - Available commands:"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make build       - build Docker image"
	@echo "  make rebuild     - rebuild Docker image without cache"
	@echo "  make shell       - open interactive shell in container"
	@echo "  make ps          - show compose services"
	@echo "  make logs        - show compose logs"
	@echo "  make gpu-check   - verify CUDA visibility inside container"
	@echo ""
	@echo "🎬 Demo:"
	@echo "  make demo        - run one test example for MODEL=$(MODEL)"
	@echo "  make demo-small  - run one smaller/faster example for MODEL=$(MODEL)"
	@echo "  make demo-svg    - run one example and save animation to $(SVG_OUTPUT_DIR)"
	@echo "  make gif-tools   - install SVG->GIF tools into persistent container $(CONTAINER_NAME)"
	@echo "  make gif         - convert $(SVG_OUTPUT_DIR)/anim_0.svg to outputs/anim_$(MODEL).gif inside container"
	@echo "  make gif-compare - combine MAGAT and HMAGAT GIFs side-by-side"
	@echo "  make gif-compare-small - combine MAGAT and HMAGAT GIFs into a lighter file"
	@echo ""
	@echo "⚙️  Config:"
	@echo "  NVIDIA_VISIBLE_DEVICES=$(NVIDIA_VISIBLE_DEVICES)"
	@echo "  CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"
	@echo "  CONTAINER_NAME=$(CONTAINER_NAME)"
	@echo "  MODEL=$(MODEL)"

build:
	@echo "🔨 Building Docker image..."
	$(DOCKER_COMPOSE) build
	@echo "✅ Image built"

rebuild:
	@echo "🔨 Rebuilding Docker image without cache..."
	$(DOCKER_COMPOSE) build --no-cache
	@echo "✅ Image rebuilt"

shell:
	@echo "🐚 Opening shell in container..."
	NVIDIA_VISIBLE_DEVICES=$(NVIDIA_VISIBLE_DEVICES) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(DOCKER_COMPOSE) run --rm $(SERVICE) bash

ps:
	@echo "📦 Compose services:"
	$(DOCKER_COMPOSE) ps

logs:
	@echo "📋 Compose logs:"
	$(DOCKER_COMPOSE) logs -f

gpu-check:
	@echo "🧪 Checking CUDA inside container..."
	NVIDIA_VISIBLE_DEVICES=$(NVIDIA_VISIBLE_DEVICES) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(DOCKER_COMPOSE) run --rm $(SERVICE) \
	python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

demo:
	@echo "🚀 Running one $(MODEL) demo example..."
	NVIDIA_VISIBLE_DEVICES=$(NVIDIA_VISIBLE_DEVICES) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(DOCKER_COMPOSE) run --rm $(SERVICE) \
	python test_imitation_learning_pyg.py $(DEMO_ARGS)

demo-small:
	@echo "⚡ Running one smaller $(MODEL) demo example..."
	NVIDIA_VISIBLE_DEVICES=$(NVIDIA_VISIBLE_DEVICES) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(DOCKER_COMPOSE) run --rm $(SERVICE) \
	python test_imitation_learning_pyg.py \
	$(DEMO_ARGS) \
	--test_name one_demo_small \
	--test_num_agents 16+16 \
	--test_max_episode_steps 192

demo-svg:
	@echo "🎬 Running one $(MODEL) demo example with SVG animation..."
	@mkdir -p $(SVG_OUTPUT_DIR)
	NVIDIA_VISIBLE_DEVICES=$(NVIDIA_VISIBLE_DEVICES) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(DOCKER_COMPOSE) run --rm $(SERVICE) \
	python test_imitation_learning_pyg.py $(DEMO_ARGS) --svg_save_dir $(SVG_OUTPUT_DIR)
	@echo "✅ Animation saved to $(SVG_OUTPUT_DIR)"

gif-tools:
	@echo "🧰 Installing SVG->GIF tools into container $(CONTAINER_NAME)..."
	docker exec $(CONTAINER_NAME) bash -lc 'apt-get update && apt-get install -y libnss3 libxss1 libasound2 libatk-bridge2.0-0 libgtk-3-0 libgbm1 fonts-liberation libxshmfence1 libglu1-mesa && python -m pip install --no-cache-dir pyppeteer && python - <<\"PY\"\nfrom pyppeteer import chromium_downloader\nchromium_downloader.download_chromium()\nprint(chromium_downloader.chromium_executable())\nPY'
	@echo "✅ GIF tools installed in $(CONTAINER_NAME)"

gif:
	@echo "🎞️  Converting SVG animation to GIF inside container $(CONTAINER_NAME)..."
	docker exec $(CONTAINER_NAME) bash -lc 'python /workspace/docker/render_svg_to_gif.py --input $(SVG_INPUT) --output $(GIF_OUTPUT) --frames-dir $(GIF_FRAMES_DIR) --fps $(GIF_FPS) --label-mode both --label-prefix "$(MODEL_LABEL)"'
	@echo "✅ GIF saved to outputs/anim_$(MODEL).gif"


gif-compare:
	@echo "🪄 Building synchronized comparison GIF..."
	docker exec $(CONTAINER_NAME) bash -lc 'python /workspace/docker/render_compare_gif.py --left /workspace/outputs/anim_magat.gif --right /workspace/outputs/anim_hmagat.gif --output $(COMPARE_GIF_OUTPUT) --left-label "MAGAT" --right-label "HMAGAT" --max-width $(COMPARE_GIF_MAX_WIDTH)'
	@echo "✅ Comparison GIF saved to outputs/anim_compare.gif"

gif-compare-small:
	@echo "🪄 Building lightweight synchronized comparison GIF..."
	docker exec $(CONTAINER_NAME) bash -lc 'python /workspace/docker/render_compare_gif.py --left /workspace/outputs/anim_magat.gif --right /workspace/outputs/anim_hmagat.gif --output $(COMPARE_GIF_SMALL_OUTPUT) --left-label "MAGAT" --right-label "HMAGAT" --max-width $(COMPARE_GIF_SMALL_MAX_WIDTH)'
	@echo "✅ Lightweight comparison GIF saved to outputs/anim_compare_small.gif"
