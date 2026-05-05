# HMAGAT-CS implementation notes

## Основная идея

Исходный `HMAGAT` хорошо отвечает на вопрос "какие агенты сейчас важны" за счет
directed hypergraph interaction. Но модель в основном остается snapshot-based:
она принимает текущее наблюдение и текущую структуру взаимодействий, после чего
сразу предсказывает действие.

В плотных MAPF-сценариях часть решений зависит не только от текущего snapshot,
но и от недавней истории конфликта:

- кто уже начал уступать в узком коридоре;
- не началась ли oscillation;
- находится ли локальная группа в фазе ожидания, прохода или восстановления;
- нужно ли продолжать уже начатый coordination pattern.

`coordination-state token` должен дать модели компактную память о фазе локального
разрешения конфликта. Интуиция:

> Interaction structure tells who matters now; coordination state tells what
> coordination regime is unfolding.

## Что имплементируем

Мы имплементируем первый минимальный вариант `HMAGAT-CS`: расширение текущей
модели `HMAGAT` / `DirectionalHMAGAT` рекуррентным coordination-state token.

Чтобы было явно видно, что именно меняется, используем единые обозначения:

- $o_i^t$ -- локальное наблюдение агента $i$ на шаге $t$;
- $z_i^t$ -- embedding после CNN encoder;
- $S_t$ -- структура взаимодействий на шаге $t$: graph для `MAGAT` или
  directed hypergraph для `DirectionalHMAGAT`;
- $h_i^t$ -- представление агента после `MAGAT` или `DirectionalHMAGAT`, уже с
  учетом graph/hypergraph interaction;
- $m_i^t$ -- новый coordination-state token агента;
- $\mathrm{logits}_i^t$ -- выход action decoder по 5 действиям.

Базовая модель до изменений:

$$
z_i^t = \mathrm{CNN}(o_i^t)
$$

$$
h_i^t = \mathrm{MAGAT/HMAGAT}(z_i^t, S_t)
$$

$$
\mathrm{logits}_i^t = \mathrm{MLP}(h_i^t)
$$

То есть до изменений action decoder видел только текущее представление
$h_i^t$, построенное по текущему snapshot.

Что именно хотим добавить на уровне агентов:

- coordination state не является одним общим global token для всей команды;
- у каждого агента $i$ есть свой собственный token $m_i^t$;
- этот token обновляется на каждом timestep-е из текущего представления того же
  агента $h_i^t$ и его собственного прошлого состояния $m_i^{t-1}$;
- поэтому $m_i^t$ можно понимать как локальную память агента о том, в каком
  coordination regime он находится: проходит, уступает, ждет, восстанавливается
  после конфликта или рискует попасть в oscillation;
- взаимодействие между агентами по-прежнему происходит через `MAGAT` /
  `DirectionalHMAGAT`, то есть через $S_t$ при вычислении $h_i^t$;
- в текущем MVP сами coordination-state tokens напрямую друг с другом не
  обмениваются.

Иными словами, `HMAGAT-CS` добавляет не глобальную память сцены, а per-agent
recurrent memory:

$$
M^t =
\begin{bmatrix}
(m_1^t)^\top \\
(m_2^t)^\top \\
\vdots \\
(m_N^t)^\top
\end{bmatrix}
\in \mathbb{R}^{N \times d_m}
$$

В коде это соответствует `self._coordination_state`: tensor формы
`[num_agents, coordination_state_size]`, где каждая строка хранит состояние
одного агента. Это делает модель децентрализованной по decoder-у: действие
агента $i$ выбирается из его текущего embedding $h_i^t$ и его собственного
state $m_i^t$.

Важно: это не привязывает архитектуру к фиксированному числу агентов. В модели
создается одна общая `GRUCell`, и она применяется ко всем строкам batch с
одними и теми же весами:

$$
m_i^t = \mathrm{GRUCell}_{\theta}(h_i^t, m_i^{t-1}), \qquad i = 1,\dots,N
$$

Параметры $\theta$ зависят от размерности $h_i^t$ и `coordination_state_size`,
но не зависят от $N$. Число агентов влияет только на runtime shape:

$$
\begin{aligned}
h &\in \mathbb{R}^{N \times d_h}, \\
M^t &\in \mathbb{R}^{N \times d_m}, \\
\mathrm{logits} &\in \mathbb{R}^{N \times 5}.
\end{aligned}
$$

Поэтому одна и та же модель может работать с разным числом агентов. Главное
условие корректности -- стабильное соответствие строки tensor-а конкретному
агенту между timestep-ами. В rollout порядок агентов задается средой; для
sequence-aware training нужно явно сохранять соответствие
`agent_id -> hidden state`, особенно если в batch попадут несколько episode или
если порядок агентов может меняться.

Новый минимальный вариант `HMAGAT-CS`:

$$
z_i^t = \mathrm{CNN}(o_i^t)
$$

$$
h_i^t = \mathrm{MAGAT/HMAGAT}(z_i^t, S_t)
$$

$$
m_i^t = \mathrm{GRUCell}(h_i^t, m_i^{t-1})
$$

$$
\mathrm{logits}_i^t = \mathrm{MLP}([h_i^t ; m_i^t])
$$

То есть после изменений мы добавляем ровно один новый шаг между
`MAGAT/HMAGAT` и `MLP`: обновление recurrent coordination state. Decoder
теперь получает не только текущее group-conditioned представление $h_i^t$, но и
краткую историю локальной координации $m_i^t$.

В терминах pipeline:

$$
\text{До:}\quad
o_i^t \rightarrow \mathrm{CNN} \rightarrow \mathrm{MAGAT/HMAGAT}
\rightarrow \mathrm{MLP} \rightarrow \mathrm{logits}_i^t
$$

$$
\text{После:}\quad
o_i^t \rightarrow \mathrm{CNN} \rightarrow \mathrm{MAGAT/HMAGAT}
\rightarrow \mathrm{GRUCell} \rightarrow \mathrm{concat}
\rightarrow \mathrm{MLP} \rightarrow \mathrm{logits}_i^t
$$

На этом этапе мы реализуем только локальное рекуррентное состояние координации:
новых каналов коммуникации между coordination-state tokens не добавляется.

## Что ожидается

Ожидаемый эффект от минимального `HMAGAT-CS`:

- более стабильное поведение в плотных сценах;
- меньше oscillation / livelock;
- лучшее symmetry breaking в узких проходах и bottleneck-сценариях;
- более последовательное выполнение yielding / passing commitments;
- улучшение прежде всего на temporally ambiguous cases, а не обязательно на
  простых sparse-картах.

## Статус sequence-aware training

Текущий статус sequential training: код для минимального sequence-aware
обучения уже реализован. Он включается флагом `--sequence_training` и проходит
через `MAPFSequenceDataset`, `collate_mapf_sequences` и `compute_sequence_loss`.
При этом default training без этого флага по-прежнему snapshot-based: recurrent
state не переносится между snapshots и каждый forward стартует с нулевого
hidden state.

Для проверки именно temporal-memory гипотезы добавлен отдельный
sequence-aware path:

- `MAPFSequenceDataset` группирует contiguous snapshots по `graph_map_id`;
- `collate_mapf_sequences` формирует mini-batch как набор episode sequences;
- `validate_sequence_dataset_assumptions` проверяет contiguous grouping,
  consistency `first_step`, стабильное число агентов внутри episode и, если
  доступен `agent_id`, стабильность порядка агентов;
- `MAPFGraphDataset` и `MAPFHypergraphDataset` добавляют
  `agent_id = arange(num_agents)`, то есть явно фиксируют текущий контракт:
  строка `k` в snapshot соответствует агенту `k` при условии, что генератор
  dataset-а сохраняет порядок агентов между timestep-ами;
- `compute_sequence_loss` проходит по timestep-ам и переносит $m_i^t$ между
  соседними snapshots одного rollout;
- `--sequence_training` включает этот path в `train_imitation_learning_pyg.py`;
- `--sequence_detach_state` дает detached-state ablation;
- `--truncated_bptt_length` отрезает gradient через state между окнами;
- validation accuracy при `--sequence_training` считается тем же sequence
  helper-ом, а не возвращается к snapshot-only оценке.

Оставшиеся ограничения sequence-aware режима:

- он опирается на assumption, что snapshots внутри `graph_map_id` идут в
  temporal order; helper проверяет contiguous grouping и `first_step`, но не
  может доказать физический temporal order без явного `timestep`;
- порядок строк агентов должен оставаться стабильным между timestep-ами;
  стандартные `MAPFGraphDataset` / `MAPFHypergraphDataset` теперь явно
  выставляют `agent_id` как row-position marker, но это не является независимым
  ID из среды и не доказывает семантическую идентичность агента, если генератор
  сам переставит строки;
- если внешний/custom dataset не содержит `agent_id`, helper логирует
  `loguru.warning` и явно продолжает через row-order assumption;
- explicit `agent_id -> hidden state` reorder/reset еще не реализован;
- masking завершившихся агентов остается в семантике существующего
  `loss_function` / `terminated`, отдельного sequence padding loss mask сейчас
  не требуется, потому что padding snapshots не добавляются в timestep batch.

Иными словами: sequential training уже есть как рабочий MVP/code path, но еще
не закрыт как полноценный экспериментальный этап на больших datasets. Следующие
шаги относятся не к написанию базового sequence-training кода с нуля, а к
проверке assumptions, воспроизводимым прогонам, ablation и возможному усилению
state API.

Сейчас основной train loop устроен как обучение на независимых snapshots:

```python
model = model.train()
for data in train_dl:
    data = data.to(device)
    optimizer.zero_grad()

    out = model(data.x, data)
    loss = loss_function(out, data, model)

    loss.backward()
    optimizer.step()
```

В таком режиме каждый `data` уже является отдельным graph/hypergraph snapshot,
поэтому recurrent state не переносится с предыдущего timestep-а.

Реализованный sequence-aware train loop в `train_imitation_learning_pyg.py`
использует отдельный helper:

```python
model = model.train()
for sequence_batch in train_dl:
    optimizer.zero_grad()

    loss = compute_sequence_loss(
        model,
        sequence_batch,
        loss_function,
        device=device,
        detach_state=args.sequence_detach_state,
        truncated_bptt_length=args.truncated_bptt_length,
        on_step=accumulate_step_metrics,
    )
    loss.backward()
    optimizer.step()
```

Внутри `compute_sequence_loss` выполняется ключевое отличие от snapshot path:
цикл по $t$ внутри episode/batch. Helper сбрасывает state на начале sequence
batch, переводит модель в `simulation=True`, временно настраивает detach mode,
проходит по `sequence_batch.timesteps`, переносит coordination state между
соседними timestep-ами и возвращает средний loss по sequence.

## Статус shard-aware training

Для full dataset путь обучения переведен на shard-aware загрузку через
`--use_shards`. Это важно, потому что legacy single-file processed dataset
занимает десятки GiB и плохо масштабируется по памяти.

Основные компоненты:

- `hmagat/downstream_shards.py` -- общая валидация stage manifests;
- `hmagat/sharded_training_dataset.py` -- lazy training loader для processed,
  hypergraph, additional-data и position shards;
- `processed_dataset/shards/training_index.json` -- sidecar index для split и
  episode lookup.

Поведение:

- первый training run при отсутствии sidecar index строит его из processed
  shards и явно логирует это как `loguru.warning`;
- последующие runs используют sidecar без повторного полного сканирования
  processed shard payloads;
- train и validation datasets используют общий `ShardedTrainingIndex`;
- sequence training использует shard-aware episode sampler, чтобы batches не
  вызывали постоянную перезагрузку больших shards.

Full README-compatible dataset уже подготовлен в:

```text
/workspace/datasets/hmagat_cs_lacam
```

Готовы:

```text
processed_dataset/shards/
hypergraphs/shards/
additional_data/shards/
positions/shards/
```

Processed shards были сравнены с legacy single processed file:

```text
Processed legacy dataset and processed shards are semantically identical:
snapshots=946714, shards=4.
```

## Практический pipeline: dataset, audit, training, inference

Этот раздел фиксирует полный рабочий набор команд для текущего HMAGAT-CS
pipeline. Все команды запускаются через существующий контейнер
`hmagat-work`; `docker run --rm` в этом workflow не используется.

Общие договоренности:

- full dataset лежит в `/workspace/datasets/hmagat_cs_lacam`;
- логи складываются в `/workspace/datasets/hmagat_cs_lacam/logs`;
- все shard-aware этапы должны использовать один и тот же `override_name`,
  `num_samples`, `dataset_seed` и generation args;
- команды ниже явно фиксируют paper-compatible defaults, на которые раньше
  можно было случайно полагаться через CLI defaults: `dataset_seed=42`,
  `map_seed=17`, `map_types=random=0.2+maze=0.8`, `map_w_min=16`,
  `map_w_max=20`, `num_agents=16+24+32`,
  `obstacle_density_min=0.2`;
- если stage обнаруживает несовместимый manifest или вынужден fallback-нуться,
  это должно быть явно залогировано через `loguru.warning`.

### 1. Генерация dataset через shards

Создать директорию логов:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs'
```

Сгенерировать expert episodes и разметку экспертом LaCAM через shards:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && python -m hmagat.generate_expert_sharded \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --num_workers 4 \
  --poll_seconds 30 \
  --resource_monitor_interval 30 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/generate_expert_sharded.log'
```

Мониторинг generation log:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && tail -f datasets/hmagat_cs_lacam/logs/generate_expert_sharded.log'
```

Конвертировать raw expert shards в processed imitation shards:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && python -m hmagat.convert_to_imitation_dataset \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --use_lists \
  --use_shards \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/convert_to_imitation_dataset_sharded.log'
```

Сгенерировать directed hypergraphs. Важно: `generate_hypergraphs` не принимает
`--use_lists`; этот флаг здесь не передается.

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && python -m hmagat.generate_hypergraphs \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --hypergraph_comm_radius 7 \
  --hyperedge_generation_method kmeans \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --hypergraph_initial_colperc 0.1 \
  --hypergraph_final_colperc 0.1 \
  --use_shards \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/generate_hypergraphs_sharded.log'
```

Сгенерировать additional data с normalized cost-to-go:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && python -m hmagat.generate_additional_data \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --add_data_cost_to_go \
  --normalize_cost_to_go \
  --clamp_cost_to_go 1.0 \
  --use_shards \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/generate_additional_data_sharded.log'
```

Сгенерировать positions для paper-style edge features:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && python -m hmagat.generate_pos \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --use_edge_attr \
  --use_lists \
  --use_shards \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/generate_pos_sharded.log'
```

Ожидаемые shard directories после всех этапов:

```text
/workspace/datasets/hmagat_cs_lacam/raw_expert_predictions/shards/
/workspace/datasets/hmagat_cs_lacam/processed_dataset/shards/
/workspace/datasets/hmagat_cs_lacam/hypergraphs/shards/
/workspace/datasets/hmagat_cs_lacam/additional_data/shards/
/workspace/datasets/hmagat_cs_lacam/positions/shards/
```

### 2. Тестирование и audit данных

Базовые unit tests HMAGAT-CS:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && python -m unittest tests.test_hmagat_cs'
```

Отдельная быстрая проверка, что checkpoint HMAGAT-CS ведет разные
coordination states для двух агентов на модельной карте. Это
artifact-dependent test: если локального pilot checkpoint нет, unit test будет
корректно пропущен через `skipTest`.

```sh
docker exec hmagat-work bash -lc 'cd /workspace && python -m unittest tests.test_hmagat_cs.HMAGATCSTest.test_trained_cs_checkpoint_keeps_distinct_states_on_two_agent_map'
```

Shard-aware sequence audit:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && python -m hmagat.audit_sequence_dataset \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --use_lists \
  --imitation_learning_model DirectionalHMAGAT \
  --hyperedge_feature_generator magat \
  --final_feature_generator magat \
  --hypergraph_comm_radius 7 \
  --hyperedge_generation_method kmeans \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --hypergraph_initial_colperc 0.1 \
  --hypergraph_final_colperc 0.1 \
  --add_data_cost_to_go \
  --normalize_cost_to_go \
  --clamp_cost_to_go 1.0 \
  --validation_fraction 0.15 \
  --test_fraction 0.15 \
  --use_shards \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/audit_sequence_dataset_sharded_015.log'
```

Ожидаемый warning:

```text
Sequence audit cannot prove physical temporal order without an explicit timestep field.
```

Это не ошибка pipeline. Он означает, что dataset содержит contiguous
`graph_map_id` order, но не содержит отдельного explicit `timestep`, которым
можно было бы строго доказать физический temporal order.

Если legacy single processed file еще сохранен, можно проверить, что processed
shards семантически совпадают с ним:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && python -m hmagat.compare_processed_dataset_shards \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --use_lists'
```

Проверенный результат для текущего full dataset:

```text
Processed legacy dataset and processed shards are semantically identical:
snapshots=946714, shards=4.
```

### 3. Запуск обучения: с нуля и с checkpoint

Текущий training loader поддерживает `--use_shards`, lazy shard loading и общий
sidecar split index. Первый запуск может построить
`processed_dataset/shards/training_index.json`; это явный logged fallback:

```text
Sharded training sidecar index is missing; rebuilding it from processed shards...
```

Такой warning допустим только при первом запуске или после удаления sidecar
index. Последующие runs должны переиспользовать sidecar.

HMAGAT-CS sequence training с нуля, pilot/smoke-режим на 1 epoch. Эта команда
проверяет, что sharded training path работает end-to-end; ее не следует
выдавать за полноценное обучение с нуля.

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && CUDA_VISIBLE_DEVICES=0 python -m hmagat.train_imitation_learning_pyg \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --use_lists \
  --imitation_learning_model DirectionalHMAGAT \
  --cnn_mode ResNetLarge_withMLP \
  --model_residuals all \
  --hyperedge_feature_generator magat \
  --final_feature_generator magat \
  --hypergraph_comm_radius 7 \
  --hyperedge_generation_method kmeans \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --hypergraph_initial_colperc 0.1 \
  --hypergraph_final_colperc 0.1 \
  --add_data_cost_to_go \
  --normalize_cost_to_go \
  --clamp_cost_to_go 1.0 \
  --use_edge_attr \
  --use_edge_attr_for_messages positions+manhattan \
  --edge_attr_cnn_mode MLP \
  --load_positions_separately \
  --coordination_state_size 32 \
  --sequence_training \
  --truncated_bptt_length 8 \
  --use_shards \
  --validation_fraction 0.15 \
  --test_fraction 0.15 \
  --num_epochs 1 \
  --batch_size 20 \
  --initial_val_size 2 \
  --validation_every_epochs 1 \
  --device -1 \
  --no-validate_sequence_training_dataset \
  --checkpoints_dir checkpoints/hmagat_cs_sequence_32_from_scratch_1ep_bs20 \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/train_hmagat_cs_sequence_32_from_scratch_1ep_bs20.log'
```

Полный full-dataset запуск с нуля, без загрузки pretrained checkpoint:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && CUDA_VISIBLE_DEVICES=0 python -m hmagat.train_imitation_learning_pyg \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --use_lists \
  --imitation_learning_model DirectionalHMAGAT \
  --cnn_mode ResNetLarge_withMLP \
  --model_residuals all \
  --hyperedge_feature_generator magat \
  --final_feature_generator magat \
  --hypergraph_comm_radius 7 \
  --hyperedge_generation_method kmeans \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --hypergraph_initial_colperc 0.1 \
  --hypergraph_final_colperc 0.1 \
  --add_data_cost_to_go \
  --normalize_cost_to_go \
  --clamp_cost_to_go 1.0 \
  --use_edge_attr \
  --use_edge_attr_for_messages positions+manhattan \
  --edge_attr_cnn_mode MLP \
  --load_positions_separately \
  --coordination_state_size 32 \
  --sequence_training \
  --truncated_bptt_length 8 \
  --use_shards \
  --validation_fraction 0.15 \
  --test_fraction 0.15 \
  --num_epochs 1 \
  --batch_size 20 \
  --initial_val_size 2 \
  --validation_every_epochs 1 \
  --device -1 \
  --checkpoints_dir checkpoints/hmagat_cs_sequence_32_from_scratch_full_1ep_bs20 \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/train_hmagat_cs_sequence_32_from_scratch_full_1ep_bs20.log'
```

Этот вариант проходит весь train split за epoch, не использует
`--load_partial_parameters_path`, не ограничивает batches через
`--max_train_batches` / `--max_validation_batches` и не отключает sequence
dataset validation. `--num_epochs 1` здесь означает один полный проход по
dataset; для реального from-scratch обучения нужно увеличить число epoch и
переименовать `checkpoints_dir` / log file соответственно.

Пока full from-scratch run не является основным планом эксперимента: приоритет
-- CS warm-start от pretrained HMAGAT и последующая inference-проверка против
baseline.

HMAGAT-CS sequence training с warm-start из pretrained HMAGAT checkpoint,
диагностический pilot на 800 batches:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && CUDA_VISIBLE_DEVICES=0 python -m hmagat.train_imitation_learning_pyg \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --use_lists \
  --imitation_learning_model DirectionalHMAGAT \
  --cnn_mode ResNetLarge_withMLP \
  --model_residuals all \
  --hyperedge_feature_generator magat \
  --final_feature_generator magat \
  --hypergraph_comm_radius 7 \
  --hyperedge_generation_method kmeans \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --hypergraph_initial_colperc 0.1 \
  --hypergraph_final_colperc 0.1 \
  --add_data_cost_to_go \
  --normalize_cost_to_go \
  --clamp_cost_to_go 1.0 \
  --use_edge_attr \
  --use_edge_attr_for_messages positions+manhattan \
  --edge_attr_cnn_mode MLP \
  --load_positions_separately \
  --coordination_state_size 32 \
  --load_partial_parameters_path checkpoints/hmagat/best.pt \
  --sequence_training \
  --truncated_bptt_length 8 \
  --use_shards \
  --validation_fraction 0.15 \
  --test_fraction 0.15 \
  --max_train_batches 800 \
  --max_validation_batches 200 \
  --num_epochs 1 \
  --batch_size 20 \
  --initial_val_size 2 \
  --validation_every_epochs 1 \
  --device -1 \
  --no-validate_sequence_training_dataset \
  --checkpoints_dir checkpoints/hmagat_cs_sequence_32_pilot_800b_1ep_bs20_gpu_resnet_residuals_all_widened \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/train_hmagat_cs_sequence_32_pilot_800b_1ep_bs20_gpu_resnet_residuals_all_widened.log'
```

Важная интерпретация: unrestricted warm-start fine-tune может разрушать
качество pretrained checkpoint даже после корректного widened decoder load.
Поэтому эта команда годится как диагностический pilot, но не как финальный
go/no-go критерий для HMAGAT-CS.

Планируемый более щадящий режим -- warmup с замороженным baseline и обучением
только CS-пути. Команда ниже станет валидной после реализации флага
`--cs_warmup_freeze_baseline`; до этого ее запускать нельзя. Поэтому блок
ниже является design note, а не runnable command.

```text
docker exec hmagat-work bash -lc 'cd /workspace && mkdir -p datasets/hmagat_cs_lacam/logs && CUDA_VISIBLE_DEVICES=0 python -m hmagat.train_imitation_learning_pyg \
  --dataset_dir /workspace/datasets/hmagat_cs_lacam \
  --override_name hmagat_cs_lacam \
  --obs_radius 5 \
  --num_samples 30000 \
  --dataset_seed 42 \
  --map_seed 17 \
  --map_types random=0.2+maze=0.8 \
  --map_w_min 16 \
  --map_w_max 20 \
  --num_agents 16+24+32 \
  --obstacle_density_min 0.2 \
  --save_termination_state \
  --expert_algorithm LaCAM \
  --obstacle_density_max 0.7 \
  --ensure_grid_config_is_generatable \
  --use_lists \
  --imitation_learning_model DirectionalHMAGAT \
  --cnn_mode ResNetLarge_withMLP \
  --model_residuals all \
  --hyperedge_feature_generator magat \
  --final_feature_generator magat \
  --hypergraph_comm_radius 7 \
  --hyperedge_generation_method kmeans \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --hypergraph_initial_colperc 0.1 \
  --hypergraph_final_colperc 0.1 \
  --add_data_cost_to_go \
  --normalize_cost_to_go \
  --clamp_cost_to_go 1.0 \
  --use_edge_attr \
  --use_edge_attr_for_messages positions+manhattan \
  --edge_attr_cnn_mode MLP \
  --load_positions_separately \
  --coordination_state_size 32 \
  --load_partial_parameters_path checkpoints/hmagat/best.pt \
  --sequence_training \
  --truncated_bptt_length 8 \
  --use_shards \
  --validation_fraction 0.15 \
  --test_fraction 0.15 \
  --max_train_batches 800 \
  --max_validation_batches 200 \
  --num_epochs 1 \
  --batch_size 20 \
  --lr_start 1e-4 \
  --lr_end 1e-5 \
  --initial_val_size 2 \
  --validation_every_epochs 1 \
  --device -1 \
  --no-validate_sequence_training_dataset \
  --cs_warmup_freeze_baseline \
  --checkpoints_dir checkpoints/hmagat_cs_sequence_32_warmup_freeze_800b_1ep_bs20 \
  2>&1 | tee datasets/hmagat_cs_lacam/logs/train_hmagat_cs_sequence_32_warmup_freeze_800b_1ep_bs20.log'
```

### 4. Inference и сравнение с baseline

Сравнение с pretrained baseline нужно делать тем же demo protocol, что
использовался раньше для HMAGAT demo: `warehouse=1.0`, `32+32` agents,
`max_episode_steps=256`, `collision_shielding=pibt`,
`action_sampling=probabilistic`, RL temperature sampler epoch `43`.

Baseline HMAGAT:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && OUT=outputs/eval_warehouse128_hmagat_vs_hmagat_cs/hmagat_baseline && mkdir -p "$OUT/svg" && CUDA_VISIBLE_DEVICES=0 python test_imitation_learning_pyg.py \
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
  --test_name warehouse128 \
  --test_num_samples 128 \
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
  --test_min_dist 10 \
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
  --temperature_sampling_model_epoch_num 43 \
  --svg_save_dir "$OUT/svg" \
  2>&1 | tee "$OUT/eval.log"'
```

HMAGAT-CS trained checkpoint:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && OUT=outputs/eval_warehouse128_hmagat_vs_hmagat_cs/hmagat_cs_sequence_32_pilot_800b_1ep_bs20_gpu_resnet_residuals_all_widened && mkdir -p "$OUT/svg" && CUDA_VISIBLE_DEVICES=0 python test_imitation_learning_pyg.py \
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
  --test_name warehouse128 \
  --test_num_samples 128 \
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
  --test_min_dist 10 \
  --hypergraph_comm_radius 7 \
  --hyperedge_generation_method kmeans \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --hypergraph_initial_colperc 0.1 \
  --hypergraph_final_colperc 0.1 \
  --checkpoints_dir checkpoints/hmagat_cs_sequence_32_pilot_800b_1ep_bs20_gpu_resnet_residuals_all_widened \
  --run_name hmagat_cs_sequence_32_pilot_800b_1ep_bs20_gpu_resnet_residuals_all_widened \
  --imitation_learning_model DirectionalHMAGAT \
  --hyperedge_feature_generator magat \
  --final_feature_generator magat \
  --coordination_state_size 32 \
  --model_epoch_num 0 \
  --rl_based_temperature_sampling \
  --temperature_checkpoints_dir checkpoints/hmagat_temperature_module \
  --temperature_run_name simple_rl \
  --temperature_actor_critic simple-local-val-init \
  --temperature_optimize only-all-on-goal \
  --iterations_per_epoch 3 \
  --temperature_min_val 0.5 \
  --temperature_max_val 0.9 \
  --temperature_sampling_model_epoch_num 43 \
  --svg_save_dir "$OUT/svg" \
  2>&1 | tee "$OUT/eval.log"'
```

Zero-shot CS parity check: создать CS-архитектуру, загрузить pretrained HMAGAT
через widened partial load, не обучать, прогнать тот же inference protocol. Если
zero-shot CS совпадает с baseline, значит архитектура/inference path корректны,
а деградация trained checkpoint относится к training strategy.

В этой команде `--load_partial_parameters_path checkpoints/hmagat/best.pt`
является фактическим источником весов основной модели. `--checkpoints_dir`
оставлен как совместимый аргумент test runner-а, но при заданном
`--load_partial_parameters_path` обычная загрузка из `checkpoints_dir/best.pt`
не используется.

```sh
docker exec hmagat-work bash -lc 'cd /workspace && OUT=outputs/eval_warehouse128_hmagat_vs_hmagat_cs/hmagat_cs_zero_shot_from_hmagat && mkdir -p "$OUT/svg" && CUDA_VISIBLE_DEVICES=0 python test_imitation_learning_pyg.py \
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
  --test_name warehouse128 \
  --test_num_samples 128 \
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
  --test_min_dist 10 \
  --hypergraph_comm_radius 7 \
  --hyperedge_generation_method kmeans \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --hypergraph_initial_colperc 0.1 \
  --hypergraph_final_colperc 0.1 \
  --checkpoints_dir checkpoints/hmagat \
  --run_name hmagat_cs_zero_shot_from_hmagat \
  --imitation_learning_model DirectionalHMAGAT \
  --hyperedge_feature_generator magat \
  --final_feature_generator magat \
  --coordination_state_size 32 \
  --load_partial_parameters_path checkpoints/hmagat/best.pt \
  --rl_based_temperature_sampling \
  --temperature_checkpoints_dir checkpoints/hmagat_temperature_module \
  --temperature_run_name simple_rl \
  --temperature_actor_critic simple-local-val-init \
  --temperature_optimize only-all-on-goal \
  --iterations_per_epoch 3 \
  --temperature_min_val 0.5 \
  --temperature_max_val 0.9 \
  --temperature_sampling_model_epoch_num 43 \
  --svg_save_dir "$OUT/svg" \
  2>&1 | tee "$OUT/eval.log"'
```

Сравнить итоговые метрики:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && grep -E "Final results|Success Rate|Average Makespan|Average Partial Success Rate|Average Sum of Costs|Testing Graph" outputs/eval_warehouse128_hmagat_vs_hmagat_cs/*/eval.log'
```

## План реализации

- [x] Добавить CLI-флаги для включения coordination state:
  - `--coordination_state_size`;
  - `--coordination_state_update`.

- [x] Расширить `DecentralPlannerGATNet`:
  - принять новые параметры в `__init__`;
  - создать `torch.nn.GRUCell`, если `coordination_state_size > 0`;
  - увеличить вход action decoder с `h_dim` до `h_dim + coordination_state_size`;
  - сохранить `cnn-to-out` residual в размерности `h_dim`, потому что residual
    складывается с выходом GNN/HGNN до добавления coordination state;
  - сбрасывать hidden state при входе/выходе из simulation.

- [x] Встроить update state в `forward` после `self.gnn(...)` и после optional
      `cnn-to-out` residual, но до `actionsMLP`.
- [x] Сохранить backward compatibility:
  - default `coordination_state_size=0`;
  - при default-флагах старая архитектура не меняется;
  - старые checkpoints должны продолжать грузиться в старую архитектуру без
    новых флагов.

- [x] Добавить partial checkpoint loading для старого HMAGAT checkpoint:
  - загрузить совместимые CNN/HGNN/decoder параметры;
  - для widened first decoder layer копировать checkpoint prefix в старые
    колонки `actionsMLP.0.weight[:, :old_in]`;
  - новые CS-колонки `actionsMLP.0.weight[:, old_in:]` инициализировать нулями;
  - загружать `actionsMLP.0.bias`, если shape совпадает;
  - оставить `coordination_state_cell.*` свежим;
  - явно логировать loaded, skipped и widened keys через `loguru.warning`.

- [x] Проверить внутри Docker Compose контейнера проекта:
  - compile;
  - synthetic smoke для `MAGAT`;
  - synthetic smoke для `MAGAT + coordination_state`;
  - synthetic smoke для `DirectionalHMAGAT`;
  - synthetic smoke для `DirectionalHMAGAT + coordination_state`;
  - synthetic smoke для `CombinedModel` temperature wrapper.

- [x] Добавить sequence-aware dataset / batching:
  - `MAPFSequenceDataset`;
  - `collate_mapf_sequences`;
  - `MAPFSequenceBatch.active_episode_indices` для variable-length episode
    batches.

- [x] Добавить sequence dataset assumption validation:
  - `validate_sequence_dataset_assumptions`;
  - проверка `first_step`, contiguous `graph_map_id`, стабильного числа агентов;
  - проверка stable `agent_id`, если поле доступно;
  - `MAPFGraphDataset` / `MAPFHypergraphDataset` добавляют
    `agent_id = arange(num_agents)` как явный row-order contract;
  - `loguru.warning`, если `agent_id` отсутствует и приходится полагаться на
    row-order assumption.

- [x] Обучать recurrent state на последовательностях expert trajectories:
  - `compute_sequence_loss`;
  - `--sequence_training`;
  - full BPTT внутри sequence batch;
  - detached-state ablation через `--sequence_detach_state`;
  - truncated gradient через `--truncated_bptt_length`;
  - sequence-aware validation accuracy.

- [ ] Следующий experiment/analysis этап:
  - реализовать `--cs_warmup_freeze_baseline` через TDD, потому что
    unrestricted fine-tune всех весов разрушает pretrained baseline policy;
  - в warmup режиме заморозить CNN/GNN/старые decoder weights и обучать только
    `coordination_state_cell.*` плюс новые decoder columns
    `actionsMLP.0.weight[:, old_in:]`;
  - защитить старые decoder columns от gradient и `weight_decay`;
  - после реализации warmup режима запустить pilot на full sharded dataset;
  - повторить assumption validation на реальных datasets и сохранить результаты;
  - добавить explicit `timestep`, если текущих полей недостаточно для строгой
    проверки temporal order;
  - заменить row-position `agent_id` на persistent environment-level
    `agent_id`, если генератор dataset-а может менять порядок строк агентов;
  - добавить partial reset или reorder by `agent_id`, если dataset этого
    потребует;
  - добавить ablation: `HMAGAT`, `HMAGAT + non-recurrent token`,
    `HMAGAT-CS`;
  - провести full experiment comparison на dense / bottleneck / narrow corridor
    сценариях.

## Что уже сделано

### 1. Добавлены CLI-флаги

Файл: `hmagat/training_args.py`.

Добавлены два аргумента:

```python
    parser.add_argument(
        "--coordination_state_size",
        type=int,
        default=0,
        help=(
            "Hidden size for an optional recurrent coordination-state token. "
            "Set to 0 to disable it."
        ),
    )
    parser.add_argument(
        "--coordination_state_update",
        type=str,
        default="gru",
        choices=["gru"],
        help="Update rule for the optional coordination-state token.",
    )
```

Смысл:

- `--coordination_state_size 0` отключает новый механизм и оставляет старую
  модель;
- `--coordination_state_size 64`, например, включает recurrent token размера
  `64`;
- `--coordination_state_update gru` пока единственный поддержанный update rule.

### 2. Расширена сигнатура DecentralPlannerGATNet

Файл: `hmagat/modules/agents.py`.

В `DecentralPlannerGATNet.__init__` добавлены параметры:

```python
        coordination_state_size=0,
        coordination_state_update="gru",
```

Они добавлены с безопасными default-значениями, поэтому существующие вызовы
конструктора остаются валидными.

### 3. Добавлен GRUCell для coordination state

Файл: `hmagat/modules/agents.py`.

После создания `self.gnn` вычисляется размер выхода graph/hypergraph блока и,
если coordination state включен, создается `GRUCell`:

```python
        gnn_output_size = num_attention_heads * embedding_sizes_gnn[-1]
        self.coordination_state_size = coordination_state_size
        self.coordination_state_update = coordination_state_update
        self.coordination_state_cell = None
        self._coordination_state = None
        self.detach_coordination_state = True
        if self.coordination_state_size > 0:
            if self.coordination_state_update != "gru":
                raise ValueError(
                    "Only GRU coordination-state updates are currently supported."
                )
            self.coordination_state_cell = torch.nn.GRUCell(
                input_size=gnn_output_size,
                hidden_size=self.coordination_state_size,
            )
```

Смысл:

- `gnn_output_size` -- размер $h_i^t$;
- `self._coordination_state` хранит recurrent state во время rollout;
- `GRUCell` обновляет $m_i^t$ из текущего $h_i^t$ и предыдущего
  $m_i^{t-1}$.

### 4. Action decoder теперь учитывает coordination state

Файл: `hmagat/modules/agents.py`.

Раньше вход `actionsMLP` был равен выходу GNN/HGNN. Теперь при включенном
coordination state он увеличивается на размер токена:

```python
        actions_input_size = gnn_output_size
        if self.coordination_state_size > 0:
            actions_input_size += self.coordination_state_size

        actions_mlp_sizes = [
            actions_input_size,
            embedding_sizes_gnn[-1],
            num_classes,
        ]
```

Это реализует decoder:

$$
\mathrm{MLP}([h_i^t ; m_i^t]) \rightarrow 5\ \mathrm{action\ logits}
$$

### 5. Сохранена корректная размерность cnn-to-out residual

Файл: `hmagat/modules/agents.py`.

Если включен `module_residual="cnn-to-out"`, residual добавляется к выходу
GNN/HGNN до применения coordination state. Поэтому его размерность должна
оставаться равной `gnn_output_size`, а не расширенному входу decoder
`gnn_output_size + coordination_state_size`:

```python
        self.cnn_to_out_lin = None
        for res in module_residual:
            if res == "cnn-to-out":
                self.cnn_to_out_lin = torch.nn.Linear(
                    cnn_output_size, gnn_output_size
                )
            else:
                raise ValueError(f"Unsupported module_residual: {res}.")
```

Это сохраняет порядок:

$$
h_i^t \leftarrow h_i^t + \mathrm{ResidualCNN}(o_i^t)
$$

и только после этого:

$$
m_i^t = \mathrm{GRUCell}(h_i^t, m_i^{t-1})
$$

### 6. Добавлен reset параметров GRUCell

Файл: `hmagat/modules/agents.py`.

В `reset_parameters` добавлен reset нового recurrent-модуля:

```python
        if self.coordination_state_cell is not None:
            self.coordination_state_cell.reset_parameters()
```

Это сохраняет общий стиль класса: все trainable-модули сбрасывают параметры
через общий `reset_parameters`.

### 7. Добавлен reset hidden state при simulation boundary

Файл: `hmagat/modules/agents.py`.

Метод `in_simulation` теперь сбрасывает coordination state при переключении
режима:

```python
    def in_simulation(self, value):
        self.simulation = value
        self.reset_coordination_state()

    def reset_coordination_state(self):
        self._coordination_state = None
```

Смысл:

- при старте rollout hidden state не должен протекать из предыдущего episode;
- при выходе из simulation состояние тоже сбрасывается;
- это важно для корректной оценки на нескольких задачах подряд.

### 8. Coordination state встроен в forward

Файл: `hmagat/modules/agents.py`.

После `self.gnn(...)` и optional `cnn-to-out` residual добавлен вызов
`_apply_coordination_state`:

```python
        x = self.gnn(x, data, **gnn_input_kwargs)

        if self.cnn_to_out_lin is not None:
            res_out = self.cnn_to_out_lin(cnn_out)
            res_out = F.relu(res_out)
            x = res_out + x
        if self.coordination_state_cell is not None:
            x = self._apply_coordination_state(x)
        for lin in self.actionsMLP[:-1]:
            x = lin(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, p=0.2, training=self.training)
        x = self.actionsMLP[-1](x)
```

Здесь `x` до `_apply_coordination_state` -- это $h_i^t$, а после -- уже
$[h_i^t ; m_i^t]$.

### 9. Реализован helper \_apply_coordination_state

Файл: `hmagat/modules/agents.py`.

Код:

```python
    def _apply_coordination_state(self, x):
        if self.simulation:
            prev_state = self._coordination_state
            if (
                prev_state is None
                or prev_state.shape[0] != x.shape[0]
                or prev_state.device != x.device
                or prev_state.dtype != x.dtype
            ):
                prev_state = x.new_zeros((x.shape[0], self.coordination_state_size))
            state = self.coordination_state_cell(x, prev_state)
            if self.detach_coordination_state:
                self._coordination_state = state.detach()
            else:
                self._coordination_state = state
        else:
            prev_state = x.new_zeros((x.shape[0], self.coordination_state_size))
            state = self.coordination_state_cell(x, prev_state)
        return torch.cat([x, state], dim=-1)
```

Поведение:

- в `simulation=True` модель хранит `self._coordination_state` между шагами;
- если число агентов, device или dtype не совпали, state безопасно
  переинициализируется нулями;
- по умолчанию `detach_coordination_state=True`, поэтому rollout/demo не держит
  computation graph между шагами;
- для sequence-aware training можно временно вызвать
  `model.set_coordination_state_detach(False)`, и тогда hidden state сохраняет
  computation graph между timestep-ами внутри BPTT window;
- в `simulation=False` используется нулевой hidden state для каждого snapshot.

Это сохраняет компромисс MVP для inference и snapshot training. В
sequence-aware training helper временно отключает detach, поэтому loss на
поздних timestep-ах может обновлять параметры через recurrent state предыдущих
timestep-ов.

### 10. Новые аргументы включены в model kwargs

Файл: `hmagat/modules/agents.py`.

В список `_GNN_DEF_KEYS` добавлены:

```python
    "coordination_state_size",
    "coordination_state_update",
```

В `model_kwargs` добавлены те же параметры:

```python
            "coordination_state_size",
            "coordination_state_update",
```

Благодаря этому флаги из CLI доходят до `DecentralPlannerGATNet`.

### 11. Проброшен simulation mode через temperature wrapper

Файл: `hmagat/modules/temperature_sampling/actor_critic.py`.

`CombinedModel` используется как wrapper вокруг основной модели при
RL-based temperature sampling. Для `HMAGAT-CS` важно, чтобы вызов
`in_simulation(True)` доходил до wrapped `DecentralPlannerGATNet`; иначе
coordination state не будет сохраняться между rollout-шагами в режиме
temperature sampling.

Код:

```python
    def in_simulation(self, in_simulation):
        self.model.in_simulation(in_simulation)
```

Это сохраняет одинаковую семантику rollout для обычной модели и для модели,
обернутой temperature sampler.

### 12. Добавлен partial checkpoint loading

Файл: `hmagat/modules/agents.py`, функция `load_partial_state_dict`.

Этот helper нужен, чтобы инициализировать `HMAGAT-CS` из старого pretrained
`HMAGAT` checkpoint. Он загружает параметры с совпадающими name/shape, а также
специально поддерживает widened linear layer, когда checkpoint weight уже, чем
текущий model weight:

```text
checkpoint: actionsMLP.0.weight = (128, 128)
HMAGAT-CS:  actionsMLP.0.weight = (128, 160)
```

В этом случае helper не пропускает весь decoder layer. Он:

- создает zero tensor формы текущей модели;
- копирует checkpoint weight в prefix `[:, :old_in]`;
- оставляет новые CS-колонки `[:, old_in:]` нулевыми;
- загружает `actionsMLP.0.bias`, если shape совпадает;
- оставляет `coordination_state_cell.*` в текущей инициализации.

Кодовая логика:

```python
def load_partial_state_dict(model, state_dict, print_prefix=""):
    model_state = model.state_dict()
    loadable_state = OrderedDict()
    skipped_missing = []
    skipped_shape = []
    skipped_related = []
    widened_linear = []
    shape_mismatch_modules = set()
    ...
    missing_after_load, unexpected_after_load = model.load_state_dict(
        loadable_state, strict=False
    )
```

Раньше widened `actionsMLP.0.weight` считался обычным shape mismatch, из-за
чего пропускался также `actionsMLP.0.bias`. Это оказалось плохим warm-start:
CS pilot стартовал с переинициализированным первым decoder layer и быстро
разрушал pretrained policy. Текущая реализация считает widened linear handled
case, логирует его как `widened linear keys: 1` и сохраняет baseline decoder
prefix.

Файлы training pipeline:

- `hmagat/train_imitation_learning_pyg.py`;
- `hmagat/post_train_quality_imp.py`.

Файл inference/demo pipeline:

- `test_imitation_learning_pyg.py` использует тот же helper, чтобы можно было
  запускать `HMAGAT-CS` rollout/demo из старого HMAGAT checkpoint без strict
  shape mismatch.

Во всех этих файлах теперь используется существующий CLI-аргумент:

```text
--load_partial_parameters_path /path/to/old_hmagat_checkpoint.pt
```

Пример для старта `HMAGAT-CS` из старого HMAGAT checkpoint:

```sh
python -m hmagat.train_imitation_learning_pyg \
  ... \
  --imitation_learning_model DirectionalHMAGAT \
  --coordination_state_size 32 \
  --load_partial_parameters_path checkpoints/hmagat/best.pt
```

Ожидаемое поведение:

- CNN/HGNN и совместимые слои загружаются из старого checkpoint;
- `coordination_state_cell.*` остается свежим;
- расширенный `actionsMLP.0.weight` получает baseline prefix и нулевые новые
  CS-колонки;
- `actionsMLP.0.bias` загружается из checkpoint, если shape совпадает;
- все loaded/skipped/missing keys логируются через `loguru.warning`, то есть
  видны как warning-сообщения, а не как тихий fallback.

Ожидаемый warning summary для корректного warm-start:

```text
[partial-load]  skipped shape-mismatch keys: 0
[partial-load]  skipped related keys: 0
[partial-load]  widened linear keys: 1
[partial-load]    actionsMLP.0.weight: copied checkpoint prefix (128, 128)
                  into model (128, 160); zero-initialized new input columns.
```

Это поведение проверено unit tests:

- widened linear partial load копирует prefix;
- новые CS-колонки нулевые;
- bias загружается при совпадении shape;
- baseline logits и zero-shot CS logits совпадают при нулевом вкладе новых
  колонок;
- checkpoint `checkpoints/hmagat/best.pt` совместим с
  `ResNetLarge_withMLP + DirectionalHMAGAT + coordination_state_size=32`;
- runtime/inference конфигурация с `model_residuals=all` отдельно используется
  в demo-командах и 2-agent checkpoint smoke;
- неправильный `cnn_mode` отклоняется через `validate_partial_load_compatibility`
  с `loguru.warning` и `ValueError`.

Общее правило для fallback в HMAGAT-CS pipeline: если код вынужден пропустить
ключи checkpoint, перейти к legacy dataset filename или иначе продолжить через
альтернативный путь, это должно быть явно залогировано через
`loguru.warning`. Штатная инициализация coordination state нулями на первом
step-е episode не считается fallback и не логируется. Если в будущем reset
state из-за mismatch shape/device/dtype начнет происходить вне ожидаемой
границы episode, это нужно перевести в явный `loguru.warning`.

## Текущий статус warm-start экспериментов

### Zero-shot CS parity

После исправления widened partial load была проверена zero-shot CS-модель:

```text
architecture:
  DirectionalHMAGAT
  ResNetLarge_withMLP
  model_residuals=all
  coordination_state_size=32

checkpoint:
  checkpoints/hmagat/best.pt

loading:
  --load_partial_parameters_path
  no CS training
```

Inference на warehouse demo protocol был остановлен вручную до завершения, но
успел пройти:

```text
Testing Graph 37/128, Current Success Rate: 1.0
```

Вывод: CS-архитектура, runtime coordination state, widened partial load и
temperature-wrapper path сами по себе не ломают baseline behavior.

### Unrestricted CS fine-tune разрушает pretrained policy

Затем был обучен pilot checkpoint:

```text
checkpoints/hmagat_cs_sequence_32_pilot_800b_1ep_bs20_gpu_resnet_residuals_all_widened/epoch_0.pt
```

Параметры:

```text
sequence_training=True
truncated_bptt_length=8
coordination_state_size=32
load_partial_parameters_path=checkpoints/hmagat/best.pt
lr_start=1e-3
batch_size=20
max_train_batches=800
num_epochs=1
trainable=all model parameters
```

Результат на `warehouse128`:

```text
Success Rate: 0.578125
Average Makespan: 166.6875
Average Partial Success Rate: 0.947021484375
Average Sum of Costs: 1826.2421875
```

Такой checkpoint не является go/no-go по идее HMAGAT-CS. Он показывает, что
pipeline обучается, но unrestricted fine-tune всех весов заметно сдвигает
pretrained CNN/GNN/decoder и ухудшает policy, которая уже хорошо работала на
demo protocol.

### Следующий кодовый шаг: freeze/warmup training

Следующий режим должен сохранить baseline policy и дать памяти минимальный
канал влияния на logits:

```text
frozen:
  CNN
  HMAGAT/MAGAT GNN
  old decoder columns actionsMLP.0.weight[:, :old_in]
  actionsMLP.0.bias
  actionsMLP.1.*

trainable:
  coordination_state_cell.*
  new decoder columns actionsMLP.0.weight[:, old_in:]
```

Обучать только `coordination_state_cell.*` нельзя: после widened partial load
новые decoder columns нулевые, поэтому recurrent state не влияет на logits,
пока `actionsMLP.0.weight[:, old_in:]` не обучается.

Требуемая реализация:

- добавить CLI flag `--cs_warmup_freeze_baseline`;
- изменить порядок setup в `train_imitation_learning_pyg.py`:

  ```text
  get_model
  load pretrain / partial checkpoint
  validate partial compatibility
  apply CS warmup freeze and decoder-column mask
  create optimizer
  create scheduler
  train
  ```

- при включенном warmup требовать `coordination_state_size > 0`,
  `load_partial_parameters_path is not None` и widened
  `actionsMLP.0.weight`; нарушение -- `loguru.warning` + `ValueError`;
- маскировать gradient для old decoder columns;
- защитить old decoder columns от `weight_decay`, предпочтительно
  восстановлением checkpoint prefix после каждого `optimizer.step()`;
- логировать trainable/frozen параметры и decoder mask через `loguru`.

Этот режим пока не реализован в коде; он зафиксирован как следующий TDD task.

## Проверки

Проверки запускались внутри существующего Docker Compose контейнера проекта
`hmagat-work`. Контейнер был поднят командой:

```sh
docker start hmagat-work
```

Важно: `--rm` не использовался.

Проверено:

```sh
docker exec hmagat-work bash -lc \
  'python -m py_compile hmagat/modules/agents.py hmagat/training_args.py hmagat/train_imitation_learning_pyg.py hmagat/post_train_quality_imp.py'
```

Результат: compile прошел.

Добавлены unit-тесты для текущего `HMAGAT-CS` pipeline:

- файл: `tests/test_hmagat_cs.py`;
- запуск:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && python -m unittest tests.test_hmagat_cs'
```

Тесты не мокают graph/model path, а создают маленькие реальные
`DecentralPlannerGATNet` модели и реальные `torch_geometric.data.Data` объекты.
Покрываются:

- расширение decoder input при `coordination_state_size > 0`;
- сохранение и reset per-agent recurrent state в `simulation=True`;
- отсутствие сохранения hidden state в snapshot mode;
- корректный `cnn-to-out` residual до добавления coordination state;
- forward для `DirectionalHMAGAT + coordination_state`;
- widened partial checkpoint loading из старой архитектуры в новую:
  decoder prefix копируется, новые CS-колонки нулевые, bias загружается;
- zero-shot parity: baseline logits и CS logits совпадают после partial load,
  пока новые CS-колонки decoder-а нулевые;
- compatibility check для реального `checkpoints/hmagat/best.pt`;
- rejection test для несовместимого `cnn_mode`;
- `first_step` contract для `MAPFHypergraphDataset`, который нужен train loop
  для подсчета map-level counters.
- 2-agent runtime test на простой карте: checkpoint строит разные
  `_coordination_state` строки для двух агентов;
- warning-контракт для fallback-сценариев: partial loading, legacy dataset
  lookup и sequence-state fallback должны логироваться через `loguru.warning`.

Результат:

```text
Ran 36 tests
OK
```

Также внутри контейнера проверено наличие основных зависимостей:

```text
torch 1.13.1
torch_geometric 2.7.0
pogema ok
```

Synthetic smoke-тесты прошли для:

- `MAGAT`;
- `MAGAT + coordination_state`;
- `DirectionalHMAGAT`;
- `DirectionalHMAGAT + coordination_state`.

После аудита отдельно проверен случай `coordination_state_size > 0` вместе с
`module_residual="cnn-to-out"`:

- `MAGAT + coordination_state + cnn-to-out`;
- `DirectionalHMAGAT + coordination_state + cnn-to-out`.

Также проверено, что `CombinedModel` из temperature sampling пробрасывает
`in_simulation(True/False)` в базовую модель:

```text
wrapped output (4, 5)
base coordination state (4, 8)
state reset True
```

Partial checkpoint loading проверен на старом `checkpoints/hmagat/best.pt` и
новой модели с `coordination_state_size=32`.

Ожидаемый smoke-output:

```text
loaded keys: 112
skipped missing keys: 0
skipped shape-mismatch keys: 0
skipped related keys: 0
widened linear keys: 1
model keys left missing: 4
actionsMLP.0.weight: copied checkpoint prefix (128, 128)
  into model (128, 160); zero-initialized new input columns.
coordination_state_cell.weight_ih
coordination_state_cell.weight_hh
coordination_state_cell.bias_ih
coordination_state_cell.bias_hh
```

Это означает, что старая HMAGAT часть загружается, первый decoder layer
сохраняет baseline prefix, новые CS-колонки decoder-а нулевые, а новые
recurrent параметры остаются инициализированными заново.

Отдельный быстрый тест на простой карте с двумя агентами. Тест зависит от
локального pilot checkpoint; в чистом checkout без этого artifact он будет
помечен как skipped, а не failed.

```sh
docker exec hmagat-work bash -lc 'cd /workspace && python -m unittest tests.test_hmagat_cs.HMAGATCSTest.test_trained_cs_checkpoint_keeps_distinct_states_on_two_agent_map'
```

Результат:

```text
Ran 1 test
OK
```

Тест загружает checkpoint
`checkpoints/hmagat_cs_sequence_32_pilot_800b_1ep_bs20_gpu_resnet_residuals_all_widened/epoch_0.pt`,
создает асимметричную карту `7x7` с двумя агентами, строит runtime hypergraph
data и проверяет:

```text
_coordination_state.shape == [2, 32]
state[0] != state[1]
```

Вывод: одна shared `GRUCell` не означает общий hidden state. Состояние хранится
per-agent и различается при разных входах агентов.

Также выполнен минимальный inference/demo smoke через `test_imitation_learning_pyg.py`
без заранее подготовленного training dataset:

```sh
docker exec hmagat-work bash -lc 'cd /workspace && python test_imitation_learning_pyg.py ... \
  --imitation_learning_model DirectionalHMAGAT \
  --coordination_state_size 32 \
  --load_partial_parameters_path checkpoints/hmagat/best.pt \
  --test_num_samples 1 \
  --test_num_agents 16+16 \
  --test_max_episode_steps 64'
```

Smoke дошел до конца rollout без shape/load/runtime ошибок:

```text
Testing Graph 1/1, Current Success Rate: 0.0
Final results:
Success Rate: 0.0
Average Makespan: 64.0
Average Partial Success Rate: 0.0
Average Sum of Costs: 1040.0
```

Нулевой success rate в этом раннем smoke не является регрессией сам по себе:
это был инженерный one-sample runtime smoke, а не demo protocol с PIBT /
temperature sampler. После widened-load фикса старый decoder prefix уже
сохраняется из checkpoint, `GRUCell` остается свежим, а новые decoder columns
нулевые до warmup обучения.

Также выполнен маленький training smoke на временном dataset в `/tmp` внутри
контейнера:

1. `hmagat.run_expert` сгенерировал 4/4 успешных expert trajectories.
2. `hmagat.convert_to_imitation_dataset` построил processed graph snapshots.
3. `hmagat.generate_hypergraphs` построил hypergraph indices.
4. `hmagat.generate_additional_data` построил normalized cost-to-go features.
5. `hmagat.generate_pos` построил positions для edge attributes.
6. `hmagat.train_imitation_learning_pyg` был запущен на 1 epoch с
   `coordination_state_size=32` и
   `--load_partial_parameters_path checkpoints/hmagat/best.pt`.

Первый запуск smoke выявил реальный pipeline bug: `MAPFHypergraphDataset` не
передавал `first_step`, хотя train loop использует `data.first_step` и для
graph, и для hypergraph batches. Исправление добавило тот же `first_step`
contract в hypergraph dataset.

После исправления training smoke прошел:

```text
Starting Training....
Epoch 0, Mean Loss: 1.6612034440040588, Mean Accuracy: 0.1666666716337204
```

После добавления sequence-aware train path выполнен отдельный smoke на том же
tiny dataset:

```sh
docker exec hmagat-work python -m hmagat.train_imitation_learning_pyg \
  --dataset_dir /tmp/hmagat_cs_train_smoke \
  --override_name hmagat_cs_smoke \
  --num_samples 4 \
  --save_termination_state \
  --imitation_learning_model DirectionalHMAGAT \
  --hyperedge_feature_generator magat \
  --final_feature_generator magat \
  --hypergraph_comm_radius 7 \
  --hypergraph_num_updates 10 \
  --hypergraph_wait_one \
  --add_data_cost_to_go \
  --normalize_cost_to_go \
  --clamp_cost_to_go 1.0 \
  --coordination_state_size 32 \
  --load_partial_parameters_path checkpoints/hmagat/best.pt \
  --num_epochs 1 \
  --skip_validation \
  --batch_size 2 \
  --sequence_training \
  --checkpoints_dir /tmp/hmagat_cs_train_smoke/checkpoints_sequence
```

Результат:

```text
Epoch 0, Mean Loss: 1.623698115348816, Mean Accuracy: 0.1875
```

Дополнительно проверены:

- snapshot training smoke:

  ```text
  Epoch 0, Mean Loss: 1.5830744008223216, Mean Accuracy: 0.2916666567325592
  ```

- sequence training с `--sequence_detach_state`;
- sequence training с `--truncated_bptt_length 1`;
- sequence-aware validation accuracy path:

  ```text
  Validation Graph 0/1, Current Success Rate: 0.0
  Finished validation path completed without runtime errors.
  ```

После этого добавлена и проверена поддержка variable-length episode batches:
`compute_sequence_loss` использует `active_episode_indices`, чтобы переносить
hidden state только для тех episodes, которые остаются активными на следующем
timestep-е. Unit-test проверяет случай `[episode0 length=1, episode1 length=2]`
и подтверждает, что loss второго timestep-а дает gradient в observation первого
timestep-а именно для `episode1`.

Также добавлено warning-логирование через `loguru` для fallback-сценариев в
sequence helper:

- нет `_coordination_state` у модели;
- нет сохраненного state для активного episode;
- state не удалось сохранить после timestep-а;
- модель не поддерживает expected state/simulation API.

Во всех случаях shape выхода был ожидаемый:

```text
(num_agents, 5)
```

Для recurrent режима дополнительно проверено, что в simulation hidden state
создается и затем сбрасывается:

```text
state_after_step (4, 8)
state_reset True
```

## Список связанных файлов

Файлы, которые входят в рабочий набор вокруг `HMAGAT-CS` и сопутствующего
demo/runbook-контекста:

- [docker/dockerfile](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docker/dockerfile)
- [docker/dockerfile_ssil](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docker/dockerfile_ssil)
- [docs/hmagat_baseline.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat_baseline.md)
- [hmagat/modules/agents.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/modules/agents.py)
- [hmagat/imitation_dataset_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/imitation_dataset_pyg.py)
- [hmagat/modules/temperature_sampling/actor_critic.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/modules/temperature_sampling/actor_critic.py)
- [hmagat/post_train_quality_imp.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/post_train_quality_imp.py)
- [hmagat/train_imitation_learning_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/train_imitation_learning_pyg.py)
- [hmagat/training_args.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/training_args.py)
- [hmagat/sequence_training.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/sequence_training.py)
- [test_imitation_learning_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/test_imitation_learning_pyg.py)
- [tests/test_hmagat_cs.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/tests/test_hmagat_cs.py)
- [docs/hmagat-cs_implementation.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat-cs_implementation.md)
- [docs/hmagat-cs_readme.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat-cs_readme.md)
- [docs/hmagat-cs_sequence_training_plan.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat-cs_sequence_training_plan.md)
- [.gitignore](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/.gitignore)

Generated/demo artifacts, которые могут переноситься в prerelease при
необходимости:

- [outputs](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/outputs)
- [renders](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/renders)

Файлы, измененные именно в рамках текущего MVP `HMAGAT-CS`:

- [hmagat/modules/agents.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/modules/agents.py)
- [hmagat/imitation_dataset_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/imitation_dataset_pyg.py)
- [hmagat/modules/temperature_sampling/actor_critic.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/modules/temperature_sampling/actor_critic.py)
- [hmagat/post_train_quality_imp.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/post_train_quality_imp.py)
- [hmagat/train_imitation_learning_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/train_imitation_learning_pyg.py)
- [hmagat/training_args.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/training_args.py)
- [hmagat/sequence_training.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/sequence_training.py)
- [test_imitation_learning_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/test_imitation_learning_pyg.py)
- [tests/test_hmagat_cs.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/tests/test_hmagat_cs.py)
- [docs/hmagat-cs_implementation.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat-cs_implementation.md)
- [docs/hmagat-cs_readme.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat-cs_readme.md)
- [docs/hmagat-cs_sequence_training_plan.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat-cs_sequence_training_plan.md)

## Текущие ограничения реализации

1. Sequence training реализован как минимальный рабочий code path.

   Default train loop вызывает модель на отдельных snapshots:

   ```text
   out = model(data.x, data)
   ```

   Поэтому без `--sequence_training` hidden state каждый раз начинается с
   нулей. С флагом `--sequence_training` уже используется
   `MAPFSequenceDataset`, timestep-wise batching и BPTT-capable state mode.
   Добавлены full BPTT, detached-state ablation, truncated BPTT через
   `--truncated_bptt_length` и sequence-aware validation accuracy. Ограничение
   здесь не в отсутствии кода, а в том, что этот path пока проверен smoke/unit
   тестами и еще не прогнан как полноценная серия экспериментов на больших
   datasets.

2. Старые checkpoints не загружаются strict-режимом в новую архитектуру.

   Если `coordination_state_size=0`, архитектура старая. Если включить
   `coordination_state_size > 0`, появляются новые веса `GRUCell`, а вход
   decoder меняет размер. Поэтому старый checkpoint нельзя строго загрузить в
   новую архитектуру, но можно использовать `--load_partial_parameters_path`,
   чтобы перенести совместимые CNN/HGNN/decoder параметры.

3. Hidden state привязан к порядку агентов в batch.

   В rollout это соответствует текущему порядку агентов в среде. Если в будущем
   появится batching нескольких эпизодов или перестановка агентов между
   timestep-ами, понадобится явно хранить соответствие
   `agent_id -> hidden state`.

## Ближайшие следующие шаги

1. Расширить dataset/state audit с smoke dataset на полный experiment dataset:

   - минимальный audit уже прогнан на `/tmp/hmagat_cs_train_smoke` через
     `python -m hmagat.audit_sequence_dataset`;
   - результат smoke audit: train и validation splits проходят проверки
     contiguous `graph_map_id`, `first_step`, стабильного числа агентов и
     stable row-position `agent_id`;
   - повторить тот же audit на полном train/validation dataset перед запуском
     исследовательского обучения;
   - проверить, что snapshots внутри каждого `graph_map_id` действительно идут
     в temporal order, а не только contiguous order;
   - помнить, что текущий `agent_id` в стандартных datasets является
     row-position marker; если генератор dataset-а может менять порядок строк,
     нужен persistent environment-level `agent_id`;
   - если порядок может меняться, добавить явный
     `agent_id -> hidden state` reorder;
   - если появятся asynchronous episode resets внутри одного batch, добавить
     `reset_coordination_state(mask=...)` или эквивалентный partial reset.

2. Зафиксировать воспроизводимый experiment pipeline:

   - baseline `HMAGAT`;
   - snapshot-trained `HMAGAT-CS`;
   - sequence-trained `HMAGAT-CS`;
   - `HMAGAT-CS` с `coordination_state_size=32`;
   - `HMAGAT-CS` с `coordination_state_size=64`;
   - full BPTT, `--sequence_detach_state` и `--truncated_bptt_length`;
   - единые seeds, splits, checkpoints и evaluation maps.

   Dataset generation для paper-compatible сравнения вынесен в
   `docs/hmagat-cs_dataset_generation.md`. Команды экспериментов и минимальные
   smoke-результаты вынесены в `docs/hmagat-cs_experiment_runbook.md`.

3. Провести содержательные ablation/evaluation:

   - сравнить `HMAGAT` vs snapshot-trained `HMAGAT-CS`;
   - сравнить snapshot-trained `HMAGAT-CS` vs sequence-trained `HMAGAT-CS`;
   - сравнить full BPTT vs truncated BPTT vs detached-state ablation;
   - добавить `HMAGAT + non-recurrent token`, чтобы отделить эффект памяти от
     простого расширения decoder capacity;
   - success rate;
   - relative SoC;
   - makespan;
   - livelock rate;
   - oscillation frequency;
   - dense / bottleneck / narrow corridor subsets.

4. Перейти от MVP к более proposal-like архитектуре из
   `docs/HMAGAT_CS_proposal_ru.md`, если per-agent CS дает полезный сигнал:

   - добавить явный $\text{group/hyperedge embeddings} \rightarrow q_t$;
   - сравнить per-agent CS и group-summary CS;
   - проверить, помогает ли $q_t$ именно в multi-agent conflict regimes, где
     per-agent memory может быть слишком локальной;
   - рассмотреть auxiliary heads для `oscillation/livelock risk`,
     `conflict phase prediction`, `yield/pass prediction`.

5. Добавить interpretability analysis learned coordination state:

   - визуализация hidden state / clustering coordination phases;
   - анализ trajectory-level transitions: yielding, passing, waiting,
     recovery;
   - проверка, отличается ли learned state на dense, bottleneck и sparse
     сценариях.
