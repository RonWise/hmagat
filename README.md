# HMAGAT-CS README

## Что это

`HMAGAT-CS` -- расширение `HMAGAT` / `DirectionalHMAGAT`, которое добавляет
per-agent recurrent coordination state после graph/hypergraph блока и перед
action decoder.

Базовый pipeline:

$$
o_i^t \rightarrow \mathrm{CNN} \rightarrow \mathrm{MAGAT/HMAGAT}
\rightarrow \mathrm{MLP} \rightarrow \mathrm{logits}_i^t
$$

`HMAGAT-CS` pipeline:

$$
h_i^t = \mathrm{MAGAT/HMAGAT}(\mathrm{CNN}(o_i^t), S_t)
$$

$$
m_i^t = \mathrm{GRUCell}(h_i^t, m_i^{t-1})
$$

$$
\mathrm{logits}_i^t = \mathrm{MLP}([h_i^t ; m_i^t])
$$

Для каждого агента хранится свой hidden state:

```text
state: [num_agents, coordination_state_size]
```

Это не global token на всю сцену. Одна общая `GRUCell` применяется ко всем
агентам с shared weights, поэтому параметры модели не привязаны к числу
агентов. Важно сохранять стабильное соответствие строки tensor-а конкретному
агенту между соседними timestep-ами rollout.

## Как включить

Основные CLI-флаги:

```text
--coordination_state_size 32
--coordination_state_update gru
```

`--coordination_state_size 0` отключает CS и оставляет старую архитектуру.
`gru` пока единственный поддержанный update rule.

Sequence-aware training включается отдельно:

```text
--sequence_training
--truncated_bptt_length 8
```

Дополнительные флаги sequence path:

```text
--sequence_detach_state
--no-validate_sequence_training_dataset
```

По умолчанию sequence dataset валидируется. Отключать проверку стоит только для
уже проверенного большого датасета.

## Текущий статус

Реализовано:

- `HMAGAT-CS` forward path для graph и hypergraph моделей;
- per-agent recurrent state в rollout;
- sequence-aware training path с BPTT / truncated BPTT;
- sequence-aware validation accuracy;
- sharded dataset pipeline от expert episodes до training loader;
- sidecar index для sharded training dataset;
- audit sequence dataset;
- partial checkpoint loading из старого `HMAGAT` checkpoint в widened
  `HMAGAT-CS` модель.

Не является частью текущей реализации:

- explicit timestep field в dataset;
- persistent environment agent id: текущий `agent_id` -- row-position marker;
- freeze/warmup режим `--cs_warmup_freeze_baseline`. Он описан в implementation
  runbook как следующий TDD task, но в коде пока не реализован.

## Загрузка из старого HMAGAT checkpoint

Старый `HMAGAT` checkpoint нельзя strict-загрузить в `HMAGAT-CS`, потому что
появляется новый `coordination_state_cell.*`, а первый decoder layer получает
больший input:

```text
h_dim + coordination_state_size
```

Для warm-start используется:

```text
--load_partial_parameters_path checkpoints/hmagat/best.pt
```

Текущий partial loader делает не обычный skip widened decoder-а, а специальную
загрузку widened linear layer:

- совместимые CNN/HGNN/decoder параметры загружаются;
- `actionsMLP.0.weight[:, :old_in]` копируется из checkpoint;
- новые CS-колонки `actionsMLP.0.weight[:, old_in:]` инициализируются нулями;
- `actionsMLP.0.bias` загружается, если shape совпадает;
- `coordination_state_cell.*` остается свежим;
- все loaded/skipped/widened/missing keys логируются через `loguru.warning`.

Ожидаемый warning summary для корректного warm-start:

```text
[partial-load]  skipped shape-mismatch keys: 0
[partial-load]  skipped related keys: 0
[partial-load]  widened linear keys: 1
[partial-load]    actionsMLP.0.weight: copied checkpoint prefix (128, 128)
                  into model (128, 160); zero-initialized new input columns.
```

После такого zero-shot load `HMAGAT-CS` должен сохранять baseline behavior до
обучения CS-памяти. Это проверяется unit-тестами на совпадение baseline logits
и zero-shot CS logits при нулевом вкладе новых decoder columns.

## Датасет и обучение через шарды

Полный sharded pipeline описан в
`docs/hmagat-cs_implementation.md`, секция "Практический pipeline: dataset,
audit, training, inference".

Этапы:

1. `python -m hmagat.generate_expert_sharded` -- генерация expert episodes по
   worker shards.
2. `python -m hmagat.convert_to_imitation_dataset --use_shards` -- конвертация
   raw expert shards в processed shards.
3. `python -m hmagat.generate_hypergraphs --use_shards` -- добавление
   hypergraph-разметки.
4. `python -m hmagat.generate_additional_data --use_shards` -- добавление
   cost-to-go и related features.
5. `python -m hmagat.generate_pos --use_shards` -- генерация positional /
   edge-attribute данных.
6. `python -m hmagat.audit_sequence_dataset --use_shards` -- аудит assumptions
   для sequence training.
7. `python -m hmagat.train_imitation_learning_pyg --use_shards` -- обучение
   напрямую из processed shards.

Sharded training loader использует sidecar index:

```text
processed_dataset/shards/training_index.json
```

Если sidecar отсутствует, он перестраивается из processed shards с явным
`loguru.warning`. Это осознанный logged fallback, а не тихое поведение.

## Demo / inference

Inference/demo запускается через `test_imitation_learning_pyg.py`. Baseline
использует pretrained checkpoint из `checkpoints/hmagat`. CS zero-shot или CS
checkpoint должны запускаться с теми же demo-параметрами, что и baseline:

```text
--cnn_mode ResNetLarge_withMLP
--model_residuals all
--collision_shielding pibt
--action_sampling probabilistic
--use_edge_attr
--use_edge_attr_for_messages positions+manhattan
--edge_attr_cnn_mode MLP
--load_positions_separately
```

Для CS-архитектуры при inference обязательно задавать тот же
`--coordination_state_size`, с которым был создан checkpoint. Это параметр
архитектуры, а не только обучения.

Zero-shot parity check запускает CS-модель из baseline checkpoint через
`--load_partial_parameters_path checkpoints/hmagat/best.pt` без обучения. Если
zero-shot CS хуже baseline, это указывает на проблему в архитектуре,
state-handling или partial load. После widened-load фикса zero-shot CS не
ломает baseline behavior на warehouse demo protocol.

## Минимальные проверки

Unit tests:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && python -m unittest tests.test_hmagat_cs tests.test_sharded_expert_generation tests.test_sharded_training_dataset'
```

Покрываются:

- `MAGAT + coordination_state`;
- `DirectionalHMAGAT + coordination_state`;
- state lifecycle в `simulation=True`;
- snapshot mode без переноса hidden state;
- sequence dataset grouping/collation/loss;
- sharded expert generation helpers;
- sharded training dataset indexing/splitting;
- widened partial load и zero-shot parity.

## Docker workflow

Для этого проекта используется существующий persistent контейнер:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && ...'
```

Не использовать `docker run --rm` для рабочих команд этого pipeline.

## Артефакты

Датасеты, checkpoints, renders и outputs не являются частью compact prerelease
переноса. Они должны оставаться локальными артефактами:

```text
datasets/
checkpoints/
outputs/
renders/
```

## Где смотреть детали

Основной документ с детальными командами, аудитом параметров, training
стратегиями и inference-сравнением:

```text
docs/hmagat-cs_implementation.md
```
