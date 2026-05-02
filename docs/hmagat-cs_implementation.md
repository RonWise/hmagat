# HMAGAT-CS implementation notes

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

```text
h:      [num_agents, h_dim]
state:  [num_agents, coordination_state_size]
logits: [num_agents, 5]
```

Поэтому одна и та же модель может работать с разным числом агентов. Главное
условие корректности -- стабильное соответствие строки tensor-а конкретному
агенту между timestep-ами. В rollout порядок агентов задается средой; для
будущего sequence-aware training нужно явно сохранять соответствие
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

```text
До:     obs -> CNN -> MAGAT/HMAGAT -> MLP -> action logits
После:  obs -> CNN -> MAGAT/HMAGAT -> GRUCell -> concat -> MLP -> action logits
```

На этом этапе мы реализуем только локальное рекуррентное состояние координации:
новых каналов коммуникации между coordination-state tokens не добавляется.

## Соответствие proposal

Исходный proposal в
`docs/HMAGAT_CS_proposal_ru.md` описывает более общий исследовательский замысел:
добавить в HMAGAT latent temporal coordination state, который помогает модели
отслеживать развитие локального конфликта во времени. Текущий код реализует
первый минимальный слой этой идеи, но не всю предложенную архитектуру сразу.

Что совпадает с proposal:

- сохраняется основная мотивация: HMAGAT хорошо моделирует higher-order
  interaction в текущем snapshot, но не имеет явного temporal state;
- добавлен recurrent coordination-state token;
- update rule реализован через `GRUCell`;
- action decoder получает не только текущее представление после
  `MAGAT/HMAGAT`, но и coordination state;
- MVP не требует отдельной supervision для токена;
- ожидаемые эффекты остаются теми же: меньше oscillation/livelock, лучшее
  symmetry breaking и более устойчивое поведение в temporally ambiguous cases.

Главное отличие от proposal: в proposal используется notation с единым
`m_t` и pooled group summary `q_t`:

```text
group/hyperedge embeddings -> pooling -> q_t
m_t = GRU(m_{t-1}, q_t)
logits_t = MLP([h_t ; m_t])
```

В текущем MVP это сознательно сужено до per-agent recurrent state без
отдельного pooling-слоя:

```text
h_i^t = MAGAT/HMAGAT(z_i^t, S_t)
m_i^t = GRUCell(h_i^t, m_i^{t-1})
logits_i^t = MLP([h_i^t ; m_i^t])
```

То есть текущая реализация не строит отдельный `q_t` из hyperedge/group
embeddings. Group context попадает в coordination state косвенно: сначала
`MAGAT/HMAGAT` смешивает информацию через graph/hypergraph interaction и
формирует $h_i^t$, затем `GRUCell` обновляет состояние агента из этого
group-conditioned embedding.

Такое сужение выбрано как MVP по двум причинам:

- оно минимально вмешивается в существующий code path `CNN -> MAGAT/HMAGAT -> MLP`;
- оно сохраняет decentralized action decoding и не вводит спорный global token,
  который мог бы привязать поведение всей команды к одному shared latent.

Что пока не реализовано из proposal:

- явный pooling `group/hyperedge embeddings -> q_t`;
- non-recurrent coordination token ablation;
- auxiliary heads для `oscillation/livelock risk`, `conflict phase prediction`,
  `yield/pass prediction`;
- полноценное sequence-aware обучение recurrent state;
- отдельный interpretability analysis learned token dynamics.

Tекущая реализация соответствует основной гипотезе proposal про
latent temporal coordination state, но является более консервативным
per-agent MVP. Она не реализует proposal буквально в части `q_t`/global token,
и это важно учитывать при описании результатов: корректнее называть текущий
вариант `HMAGAT-CS MVP`, а не полной версией proposal.

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

```text
interaction structure tells who matters now
coordination state tells what coordination regime is unfolding
```

## Что ожидается

Ожидаемый эффект от минимального `HMAGAT-CS`:

- более стабильное поведение в плотных сценах;
- меньше oscillation / livelock;
- лучшее symmetry breaking в узких проходах и bottleneck-сценариях;
- более последовательное выполнение yielding / passing commitments;
- улучшение прежде всего на temporally ambiguous cases, а не обязательно на
  простых sparse-картах.

Важное ограничение текущего MVP: существующий train loop обучает модель на
отдельных graph snapshots, а не на последовательностях. Поэтому сейчас recurrent
state полноценно сохраняет историю в `simulation` / rollout, но при обычном
snapshot-training стартует с нулевого hidden state.

Для настоящего sequence-aware обучения нужен отдельный следующий этап:

- хранить в dataset границы эпизодов и порядок timestep-ов;
- формировать mini-batch как набор последовательностей, а не независимых
  snapshots;
- сбрасывать $m_i^t$ только на границах episode;
- переносить hidden state между соседними timestep-ами одного rollout;
- считать imitation loss по всем шагам последовательности;
- маскировать агентов, которые уже завершили движение или не должны
  участвовать в loss.

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

Для sequence-aware обучения train loop должен стать примерно таким:

```python
model = model.train()
for episode_batch in sequence_train_dl:
    optimizer.zero_grad()
    model.reset_coordination_state()

    total_sequence_loss = 0.0

    for data_t in episode_batch.timesteps:
        data_t = data_t.to(device)

        out_t = model(data_t.x, data_t)
        loss_t = loss_function(out_t, data_t, model)
        loss_t = mask_finished_agents(loss_t, data_t)

        total_sequence_loss = total_sequence_loss + loss_t

    total_sequence_loss.backward()
    optimizer.step()
```

То есть ключевое отличие -- цикл по $t$ внутри episode/batch. Модель получает
последовательные graph/hypergraph объекты одного rollout и обновляет
coordination state так же, как во время inference.

## План реализации

- [X] Добавить CLI-флаги для включения coordination state:

  - `--coordination_state_size`;
  - `--coordination_state_update`.
- [X] Расширить `DecentralPlannerGATNet`:

  - принять новые параметры в `__init__`;
  - создать `torch.nn.GRUCell`, если `coordination_state_size > 0`;
  - увеличить вход action decoder с `h_dim` до `h_dim + coordination_state_size`;
  - сохранить `cnn-to-out` residual в размерности `h_dim`, потому что residual
    складывается с выходом GNN/HGNN до добавления coordination state;
  - сбрасывать hidden state при входе/выходе из simulation.
- [X] Встроить update state в `forward` после `self.gnn(...)` и после optional
  `cnn-to-out` residual, но до `actionsMLP`.
- [X] Сохранить backward compatibility:

  - default `coordination_state_size=0`;
  - при default-флагах старая архитектура не меняется;
  - старые checkpoints должны продолжать грузиться в старую архитектуру без
    новых флагов.
- [X] Добавить partial checkpoint loading для старого HMAGAT checkpoint:

  - загрузить совместимые CNN/HGNN/decoder параметры;
  - оставить `GRUCell` и расширенный decoder layer свежими;
  - явно логировать loaded и skipped keys.
- [X] Проверить внутри Docker Compose контейнера проекта:

  - compile;
  - synthetic smoke для `MAGAT`;
  - synthetic smoke для `MAGAT + coordination_state`;
  - synthetic smoke для `DirectionalHMAGAT`;
  - synthetic smoke для `DirectionalHMAGAT + coordination_state`;
  - synthetic smoke для `CombinedModel` temperature wrapper.
- [ ] Следующий этап после MVP:

  - добавить sequence-aware dataset / batching;
  - обучать recurrent state на последовательностях expert trajectories;
  - добавить ablation: `HMAGAT`, `HMAGAT + non-recurrent token`,
    `HMAGAT-CS`.

## Что уже сделано

### 1. Добавлены CLI-флаги

Файл: `hmagat/training_args.py`, строки 128-143.

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

Файл: `hmagat/modules/agents.py`, строки 660-669.

В `DecentralPlannerGATNet.__init__` добавлены параметры:

```python
        coordination_state_size=0,
        coordination_state_update="gru",
```

Они добавлены с безопасными default-значениями, поэтому существующие вызовы
конструктора остаются валидными.

### 3. Добавлен GRUCell для coordination state

Файл: `hmagat/modules/agents.py`, строки 758-771.

После создания `self.gnn` вычисляется размер выхода graph/hypergraph блока и,
если coordination state включен, создается `GRUCell`:

```python
        gnn_output_size = num_attention_heads * embedding_sizes_gnn[-1]
        self.coordination_state_size = coordination_state_size
        self.coordination_state_update = coordination_state_update
        self.coordination_state_cell = None
        self._coordination_state = None
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

- `gnn_output_size` -- размер `h_i^t`;
- `self._coordination_state` хранит recurrent state во время rollout;
- `GRUCell` обновляет `m_i^t` из текущего `h_i^t` и предыдущего `m_i^{t-1}`.

### 4. Action decoder теперь учитывает coordination state

Файл: `hmagat/modules/agents.py`, строки 779-787.

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

Файл: `hmagat/modules/agents.py`, строки 800-807.

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

Файл: `hmagat/modules/agents.py`, строки 810-824.

В `reset_parameters` добавлен reset нового recurrent-модуля:

```python
        if self.coordination_state_cell is not None:
            self.coordination_state_cell.reset_parameters()
```

Это сохраняет общий стиль класса: все trainable-модули сбрасывают параметры
через общий `reset_parameters`.

### 7. Добавлен reset hidden state при simulation boundary

Файл: `hmagat/modules/agents.py`, строки 826-831.

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

Файл: `hmagat/modules/agents.py`, строки 837-877.

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

Здесь `x` до `_apply_coordination_state` -- это `h_i^t`, а после -- уже
`[h_i^t ; m_i^t]`.

### 9. Реализован helper _apply_coordination_state

Файл: `hmagat/modules/agents.py`, строки 879-894.

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
            self._coordination_state = state.detach()
        else:
            prev_state = x.new_zeros((x.shape[0], self.coordination_state_size))
            state = self.coordination_state_cell(x, prev_state)
        return torch.cat([x, state], dim=-1)
```

Поведение:

- в `simulation=True` модель хранит `self._coordination_state` между шагами;
- если число агентов, device или dtype не совпали, state безопасно
  переинициализируется нулями;
- `self._coordination_state = state.detach()` не держит computation graph между
  rollout-шагами;
- в `simulation=False` используется нулевой hidden state для каждого snapshot.

Это компромисс MVP: inference уже получает temporal memory, а training остается
совместимым с текущим snapshot-based dataset.

### 10. Новые аргументы включены в model kwargs

Файл: `hmagat/modules/agents.py`, строки 908-918.

В список `_GNN_DEF_KEYS` добавлены:

```python
    "coordination_state_size",
    "coordination_state_update",
```

Файл: `hmagat/modules/agents.py`, строки 925-937.

В `model_kwargs` добавлены те же параметры:

```python
            "coordination_state_size",
            "coordination_state_update",
```

Благодаря этому флаги из CLI доходят до `DecentralPlannerGATNet`.

### 11. Проброшен simulation mode через temperature wrapper

Файл:
`hmagat/modules/temperature_sampling/actor_critic.py`, строки 21-22.

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

Файл: `hmagat/modules/agents.py`, строки 1000-1069, функция
`load_partial_state_dict`.

Этот helper нужен, чтобы инициализировать `HMAGAT-CS` из старого pretrained
`HMAGAT` checkpoint. Он загружает только те параметры, для которых совпадают имя
и shape, а несовместимые параметры оставляет в текущей инициализации модели.

Кодовая логика:

```python
def load_partial_state_dict(model, state_dict, print_prefix=""):
    model_state = model.state_dict()
    loadable_state = OrderedDict()
    skipped_missing = []
    skipped_shape = []
    skipped_related = []
    shape_mismatch_modules = set()
    ...
    missing_after_load, unexpected_after_load = model.load_state_dict(
        loadable_state, strict=False
    )
```

Если у параметра есть shape mismatch, например:

```text
actionsMLP.0.weight: checkpoint (128, 128) -> model (128, 160)
```

то helper пропускает и остальные параметры этого же модуля, например
`actionsMLP.0.bias`. Это важно, чтобы расширенный decoder layer был
инициализирован заново целиком, а не частично.

Файлы training pipeline:

- `hmagat/train_imitation_learning_pyg.py`, import на строке 177 и вызов на
  строках 382-386;
- `hmagat/post_train_quality_imp.py`, import на строке 302 и вызов на строках
  439-443.

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
- расширенный `actionsMLP.0.*` остается свежим;
- все skipped keys печатаются явно.

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

Это означает, что старая HMAGAT часть загружается, а новые recurrent параметры и
расширенный decoder layer остаются инициализированными заново.

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

Нулевой success rate в этом smoke не является регрессией сам по себе: `GRUCell`
и расширенный первый decoder layer инициализированы заново и еще не обучались.
Цель проверки была инженерной -- подтвердить, что `HMAGAT-CS` inference path
работает с partial-loaded старым checkpoint.

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

## Список измененных файлов

Файлы, которые входят в рабочий набор изменений вокруг `HMAGAT-CS` и
сопутствующего demo/runbook-контекста:

- [docker/dockerfile](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docker/dockerfile)
- [docker/dockerfile_ssil](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docker/dockerfile_ssil)
- [docs/hmagat_baseline.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat_baseline.md)
- [hmagat/modules/agents.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/modules/agents.py)
- [hmagat/modules/temperature_sampling/actor_critic.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/modules/temperature_sampling/actor_critic.py)
- [hmagat/post_train_quality_imp.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/post_train_quality_imp.py)
- [hmagat/train_imitation_learning_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/train_imitation_learning_pyg.py)
- [hmagat/training_args.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/training_args.py)
- [test_imitation_learning_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/test_imitation_learning_pyg.py)
- [docs/hmagat-cs_implementation.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat-cs_implementation.md)

Untracked директории с generated/demo artifacts:

- [outputs](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/outputs)
- [renders](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/renders)

Файлы, измененные именно в рамках текущего MVP `HMAGAT-CS`:

- [hmagat/modules/agents.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/modules/agents.py)
- [hmagat/modules/temperature_sampling/actor_critic.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/modules/temperature_sampling/actor_critic.py)
- [hmagat/post_train_quality_imp.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/post_train_quality_imp.py)
- [hmagat/train_imitation_learning_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/train_imitation_learning_pyg.py)
- [hmagat/training_args.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/hmagat/training_args.py)
- [test_imitation_learning_pyg.py](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/test_imitation_learning_pyg.py)
- [docs/hmagat-cs_implementation.md](/home/work/WORK/MIPT/study/repos/heuristics/hmagat/docs/hmagat-cs_implementation.md)

Остальные dirty-файлы уже были в рабочем дереве до текущего implementation-log
шага и не относятся напрямую к добавлению coordination-state token.

## Текущие ограничения реализации

1. Это еще не полноценное sequence training.

   Текущий train loop вызывает модель на отдельных snapshots:

   ```text
   out = model(data.x, data)
   ```

   Поэтому при `simulation=False` hidden state каждый раз начинается с нулей.
   Чтобы обучать память как память, нужен dataset/batcher по последовательностям.
2. Старые checkpoints не загружаются strict-режимом в новую архитектуру.

   Если `coordination_state_size=0`, архитектура старая. Если включить
   `coordination_state_size > 0`, появляются новые веса `GRUCell`, а вход
   decoder меняет размер. Поэтому старый checkpoint нельзя строго загрузить в
   новую архитектуру, но можно использовать `--load_partial_parameters_path`,
   чтобы перенести совместимые CNN/HGNN/decoder параметры.
3. Hidden state привязан к порядку агентов в batch.

   В rollout это соответствует текущему порядку агентов в среде. Если в будущем
   появится batching нескольких эпизодов или перестановка агентов между
   timestep-ами, понадобится явно хранить соответствие `agent_id -> hidden state`.

## Ближайшие следующие шаги

1. Сделать маленький training smoke на небольшом dataset, чтобы проверить, что
   новый режим не ломает backward pass.
2. Добавить sequence-aware training path:

   - группировать данные по episode;
   - сохранять порядок timesteps;
   - сбрасывать hidden state на границах episode;
   - считать loss по всем шагам последовательности.
3. Добавить experiment configs:

   - baseline `HMAGAT`;
   - `HMAGAT-CS` с `coordination_state_size=32`;
   - `HMAGAT-CS` с `coordination_state_size=64`;
   - dense / bottleneck / narrow corridor evaluation.
4. После этого переходить к ablation и анализу токена:

   - livelock rate;
   - oscillation frequency;
   - success rate;
   - relative SoC;
   - визуализация hidden state / clustering coordination phases.
