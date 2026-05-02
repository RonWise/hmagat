# HMAGAT-CS README

## Что это

`HMAGAT-CS` -- минимальное расширение `HMAGAT` / `DirectionalHMAGAT`, которое
добавляет per-agent recurrent coordination state после graph/hypergraph блока и
перед action decoder.

Базовый pipeline:

```text
obs -> CNN -> MAGAT/HMAGAT -> MLP -> action logits
```

`HMAGAT-CS` pipeline:

```text
obs -> CNN -> MAGAT/HMAGAT -> GRUCell -> concat -> MLP -> action logits
```

Для каждого агента хранится свой hidden state:

```text
state: [num_agents, coordination_state_size]
```

Это не global token на всю сцену. Одна общая `GRUCell` применяется ко всем
агентам с shared weights, поэтому параметры модели не привязаны к числу
агентов. Важно только сохранять стабильное соответствие строки tensor-а
конкретному агенту между timestep-ами rollout.

## Как включить

Новые CLI-флаги:

```text
--coordination_state_size 32
--coordination_state_update gru
```

`--coordination_state_size 0` отключает CS и оставляет старую архитектуру.
`gru` пока единственный поддержанный update rule.

## Загрузка из старого HMAGAT checkpoint

Старый `HMAGAT` checkpoint нельзя strict-загрузить в `HMAGAT-CS`, потому что:

- появляется новый `coordination_state_cell.*`;
- первый decoder layer получает больший input:
  `h_dim + coordination_state_size`.

Для transfer initialization используется partial loading:

```text
--load_partial_parameters_path checkpoints/hmagat/best.pt
```

Ожидаемое поведение:

- совместимые CNN/HGNN/decoder параметры загружаются;
- `coordination_state_cell.*` инициализируется заново;
- расширенный `actionsMLP.0.*` инициализируется заново целиком;
- все skipped keys явно печатаются в log.

Типичный log:

```text
Partial checkpoint loading:
  loaded keys: 110
  skipped shape-mismatch keys: 1
  skipped related keys: 1
  model keys left missing: 6
  actionsMLP.0.weight: checkpoint (128, 128) -> model (128, 160)
  actionsMLP.0.bias
  coordination_state_cell.weight_ih
  coordination_state_cell.weight_hh
  coordination_state_cell.bias_ih
  coordination_state_cell.bias_hh
```

## Demo / inference smoke

Baseline demo не требует заранее подготовленного training dataset: сценарии
генерируются на лету в `test_imitation_learning_pyg.py`, а модель грузится из
checkpoint.

Пример минимального `HMAGAT-CS` smoke внутри существующего контейнера
`hmagat-work`:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && python test_imitation_learning_pyg.py \
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
  --test_name hmagat_cs_smoke \
  --test_num_samples 1 \
  --test_obs_radius 5 \
  --test_map_types warehouse=1.0 \
  --test_num_agents 16+16 \
  --test_wall_width_min 8 \
  --test_wall_width_max 8 \
  --test_vertical_gap 1 \
  --test_num_wall_rows_min 5 \
  --test_num_wall_rows_max 5 \
  --test_num_wall_cols_min 2 \
  --test_num_wall_cols_max 2 \
  --test_side_pad 3 \
  --test_max_episode_steps 64 \
  --test_min_dist 10 \
  --hypergraph_comm_radius 7 \
  --hyperedge_generation_method kmeans \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --hypergraph_initial_colperc 0.1 \
  --hypergraph_final_colperc 0.1 \
  --checkpoints_dir checkpoints/hmagat \
  --run_name hmagat_cs_smoke \
  --imitation_learning_model DirectionalHMAGAT \
  --hyperedge_feature_generator magat \
  --final_feature_generator magat \
  --coordination_state_size 32 \
  --load_partial_parameters_path checkpoints/hmagat/best.pt'
```

Важно: для project Docker workflow не использовать `--rm`, если задача требует
работать с существующим persistent контейнером.

Smoke считается успешным, если:

- partial loading напечатал loaded/skipped keys;
- rollout дошел до `Final results`;
- нет shape/load/runtime ошибок.

Нулевой success rate в таком smoke не является самостоятельной регрессией:
`GRUCell` и расширенный первый decoder layer свежие и еще не обучались.

## Unit-тесты

Запуск:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && python -m unittest tests.test_hmagat_cs'
```

Тесты создают маленькие реальные `DecentralPlannerGATNet` модели и реальные
`torch_geometric.data.Data` graph/hypergraph объекты. Покрываются:

- `MAGAT + coordination_state`;
- `DirectionalHMAGAT + coordination_state`;
- state lifecycle в `simulation=True`;
- snapshot mode без переноса hidden state;
- `cnn-to-out` residual до добавления CS;
- partial loading из старой архитектуры в новую.

## Training status

Текущий train loop остается snapshot-based:

```python
out = model(data.x, data)
```

Поэтому при обычном training recurrent state стартует с нулей на каждом
snapshot. Это проверяет совместимость forward/backward, но еще не обучает CS как
настоящую память последовательности.

Для полноценного sequence-aware training нужен следующий этап:

- хранить episode boundaries и timestep order;
- формировать batch как последовательности;
- сбрасывать hidden state только на границах episode;
- переносить state между соседними timestep-ами rollout;
- считать loss по всем timestep-ам;
- маскировать завершившихся агентов.

## Где смотреть детали

- Implementation notes: `docs/hmagat-cs_implementation.md`;
- Research proposal: `docs/HMAGAT_CS_proposal_ru.md`;
- Baseline demo runs: `docs/hmagat_baseline.md`.
