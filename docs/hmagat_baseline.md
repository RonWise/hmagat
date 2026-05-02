# HMAGAT baseline demo runs

## Цель документа

Этот документ фиксирует baseline demo runs для исходных моделей `MAGAT` и
`HMAGAT` до добавления `HMAGAT-CS`. Он нужен как воспроизводимая точка
сравнения: какие модели запускались, на какой карте, с какими параметрами, какие
артефакты получены и какой качественный вывод сделан.

Главный baseline result:

```text
MAGAT:  120 frames, ~20.0 s
HMAGAT: 72 frames,  ~12.0 s
```

На одном и том же плотном warehouse-сценарии `HMAGAT` завершает rollout быстрее
и визуально лучше разруливает локальные конфликты.

## Что запускалось

Запускались две pretrained модели из репозитория:

- `MAGAT`:
  - checkpoint: `checkpoints/magat`;
  - `run_name`: `magat`;
  - model class: `MAGAT`.

- `HMAGAT`:
  - checkpoint: `checkpoints/hmagat`;
  - `run_name`: `hmagat`;
  - model class: `DirectionalHMAGAT`;
  - hypergraph generation: `kmeans`;
  - temperature sampler checkpoint: `checkpoints/hmagat_temperature_module/epoch_43.pt`.

Для обеих моделей использовался один demo-сценарий:

- `test_name`: `one_demo`;
- `test_num_samples`: `1`;
- map type: `warehouse=1.0`;
- agents: `32+32`;
- observation radius: `5`;
- max episode steps: `256`;
- collision shielding: `pibt`;
- action sampling: `probabilistic`;
- CNN encoder: `ResNetLarge_withMLP`;
- additional data: normalized cost-to-go channel.

Параметры карты:

```text
--test_map_types warehouse=1.0
--test_num_agents 32+32
--test_wall_width_min 8
--test_wall_width_max 8
--test_vertical_gap 1
--test_num_wall_rows_min 5
--test_num_wall_rows_max 5
--test_num_wall_cols_min 2
--test_num_wall_cols_max 2
--test_side_pad 3
--test_max_episode_steps 256
--test_min_dist 10
```

## Docker setup

### Требования

Нужны:

- Docker;
- Docker Compose v2 (`docker compose`);
- NVIDIA Container Runtime / GPU support;
- доступная CUDA GPU;
- локальные checkpoints, уже лежащие в репозитории:
  - `checkpoints/magat/best.pt`;
  - `checkpoints/hmagat/best.pt`;
  - `checkpoints/hmagat_temperature_module/epoch_43.pt`.

### Docker image

Основной образ собирается из:

```text
docker/dockerfile
```

Базовый image:

```text
pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
```

Внутри устанавливаются:

- Python dependencies из `docker/requirements.txt`;
- `torch_geometric`;
- local wheels из `docker/wheels`;
- LaCAM/C++ dependencies (`cmake`, `g++`, `boost`, `yaml-cpp`, `libtorch`, etc.);
- `pyamg`;
- OpenCV dependencies.

Compose-конфигурация:

```text
docker/docker-compose.yml
```

Ключевые свойства compose service:

```text
service: hmagat
image: hmagat:dev
working_dir: /workspace
volume: ..:/workspace
gpus: all
PYTHONPATH=/workspace
```

### Build

Из корня репозитория:

```sh
make build
```

Проверить GPU внутри контейнера:

```sh
make gpu-check
```

Ожидаемо внутри контейнера `torch.cuda.is_available()` должен быть `True`.

## Команды запуска baseline demo

Все команды ниже запускаются из корня репозитория.

### 1. Запустить MAGAT и сохранить SVG

```sh
make demo-svg MODEL=magat
```

Ожидаемый SVG:

```text
outputs/svg_magat/anim_0.svg
```

### 2. Конвертировать MAGAT SVG в GIF

Для конвертации SVG в GIF используется persistent контейнер, потому что tools
ставятся внутрь него и вызываются через `docker exec`.

Если persistent контейнер еще не запущен:

```sh
docker compose -f docker/docker-compose.yml run -d --name hmagat-work hmagat sleep infinity
```

Установить инструменты конвертации один раз:

```sh
make gif-tools
```

Сконвертировать MAGAT:

```sh
make gif MODEL=magat
```

Ожидаемый GIF:

```text
outputs/anim_magat.gif
```

### 3. Запустить HMAGAT и сохранить SVG

```sh
make demo-svg MODEL=hmagat
```

Ожидаемый SVG:

```text
outputs/svg_hmagat/anim_0.svg
```

### 4. Конвертировать HMAGAT SVG в GIF

```sh
make gif MODEL=hmagat
```

Ожидаемый GIF:

```text
outputs/anim_hmagat.gif
```

### 5. Собрать side-by-side comparison GIF

Полная версия:

```sh
make gif-compare
```

Ожидаемый файл:

```text
outputs/anim_compare.gif
```

Легкая версия для презентации:

```sh
make gif-compare-small
```

Ожидаемый файл:

```text
outputs/anim_compare_small.gif
```

## Полные команды, которые вызывает Makefile

### Общие аргументы demo

Обе модели запускаются через `test_imitation_learning_pyg.py` с общими
аргументами:

```sh
python test_imitation_learning_pyg.py \
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
```

### MAGAT-specific arguments

```sh
--checkpoints_dir checkpoints/magat \
--run_name magat \
--imitation_learning_model MAGAT
```

### HMAGAT-specific arguments

```sh
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
```

### SVG output argument

`make demo-svg` additionally appends:

```sh
--svg_save_dir outputs/svg_magat
```

for `MODEL=magat`, and:

```sh
--svg_save_dir outputs/svg_hmagat
```

for `MODEL=hmagat`.

## Полученные артефакты

### MAGAT

```text
outputs/svg_magat/anim_0.svg
outputs/anim_magat.gif
outputs/gif_frames_magat/
```

Measured artifacts:

```text
SVG duration: 20.0 s
GIF frames:   120
GIF duration: ~19.99 s
GIF size:     591147 bytes
```

### HMAGAT

```text
outputs/svg_hmagat/anim_0.svg
outputs/anim_hmagat.gif
outputs/gif_frames_hmagat/
```

Measured artifacts:

```text
SVG duration: 12.0 s
GIF frames:   72
GIF duration: ~11.99 s
GIF size:     394984 bytes
```

### Side-by-side comparison

```text
outputs/anim_compare.gif
outputs/anim_compare_small.gif
```

Measured artifacts:

```text
outputs/anim_compare.gif:
  frames:   121
  duration: ~19.99 s
  size:     9310430 bytes

outputs/anim_compare_small.gif:
  frames:   121
  duration: ~19.99 s
  size:     7684518 bytes
```

The comparison GIF keeps the shorter rollout on its final frame after it
finishes, so both sides remain synchronized on the same timeline.

### Additional standalone demo artifact

There is also a standalone demo artifact:

```text
outputs/svg/anim_0.svg
outputs/anim_0.gif
outputs/gif_frames/
```

Measured artifacts:

```text
SVG duration: 22.5 s
GIF frames:   135
GIF duration: ~22.49 s
```

This file is useful as an extra sanity-check animation, but the main MAGAT vs.
HMAGAT comparison is based on `outputs/anim_magat.gif`,
`outputs/anim_hmagat.gif`, and `outputs/anim_compare_small.gif`.

## Result summary

For the documented warehouse demo:

| Model    | Frames | Approx. duration | Artifact                  |
| -------- | -----: | ---------------: | ------------------------- |
| `MAGAT`  |  `120` |         `20.0 s` | `outputs/anim_magat.gif`  |
| `HMAGAT` |   `72` |         `12.0 s` | `outputs/anim_hmagat.gif` |

Interpretation:

- `HMAGAT` reaches completion faster on this dense warehouse example.
- The visual comparison supports the core HMAGAT claim: explicit higher-order
  group interaction helps in dense scenarios where pairwise interaction is not
  enough.
- This is not a full benchmark; it is a reproducible qualitative demo used for
  presentation and sanity checking.
