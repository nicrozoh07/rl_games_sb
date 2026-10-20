# Agente inversor con DQN (proyecto final de Aprendizaje por Refuerzo)

Un agente de Reinforcement Learning decide cada día qué fracción de un capital
de $100.000 invertir en la acción de Alphabet (GOOGL): 0 %, 50 % o 100 %. Paga
10 bps por cada operación y se compara con comprar y mantener y con un agente
aleatorio en un período que no vio al entrenar.

Este repositorio es un fork de
[`emiliomunozai/rl_games_sb`](https://github.com/emiliomunozai/rl_games_sb): el
proyecto reutiliza su DQN (Stable-Baselines3), sus configuraciones y su
seguimiento con MLflow. La documentación original sigue más abajo.

## Estructura

| Estructura sugerida por la actividad | En este repositorio |
|---|---|
| `data/` | [`data/README.md`](data/README.md): el dataset, su versión y su SHA-256 |
| `src/environment.py` | [`investor.py`](src/rl_games_sb/investor.py): datos, división temporal, acciones y el entorno `InvestorEnv`; [`features.py`](src/rl_games_sb/features.py): los indicadores del estado |
| `src/agent.py` | DQN de Stable-Baselines3, creado por [`registry.py`](src/rl_games_sb/registry.py) con [`configs/investor_dqn_penalized.yaml`](configs/investor_dqn_penalized.yaml) |
| `src/train.py` | [`investor_cli.py`](src/rl_games_sb/investor_cli.py): `uv run investor train` |
| `src/evaluate.py` | [`evaluation.py`](src/rl_games_sb/evaluation.py): `uv run investor evaluate` |
| `src/baselines.py` | [`backtest.py`](src/rl_games_sb/backtest.py) (`buy_and_hold`, `random_policy` y las métricas): `uv run investor baselines` |
| `notebooks/eda.ipynb` | [`notebooks/investor_eda.ipynb`](notebooks/investor_eda.ipynb) y los demás notebooks (ver abajo) |
| `configs/` | `investor_dqn.yaml` (recompensa v0), `investor_dqn_penalized.yaml` (v1, agente final), `investor_final.json` (agente congelado) |

## Instalación

Requiere [uv](https://docs.astral.sh/uv/). uv instala Python 3.11 y las
versiones exactas de `uv.lock`:

```bash
uv sync
```

PyTorch se instala con soporte CUDA, pero el agente se entrena en CPU: su red
es pequeña y en CPU el resultado es reproducible bit a bit.

## Dataset

`STOCKS_GOOGL` del paquete `gym-anytrading` 2.0.0: precios diarios de GOOGL
del 2009-05-22 al 2018-08-29, 2.335 filas. No hay que descargar nada, porque
viene con `uv sync`. Origen, SHA-256 y división temporal en
[`data/README.md`](data/README.md).

| Período | Fechas | Uso |
|---|---|---|
| train | 2009-05-22 → 2014-12-31 | entrenar |
| valid | 2015-01-01 → 2015-12-31 | comparar recompensas y elegir la penalización |
| test | 2016-01-01 → 2018-08-29 | evaluación final, una sola vez |

## Entrenar

```bash
uv run investor train
```

Entrena el agente final, DQN con la recompensa v1 (log-retorno neto de costos
menos 400 bps por unidad de posición cambiada), con las semillas 0, 1 y 2.
Toma unos 2–3 minutos por semilla en CPU y guarda
`saves/dqn_investor-pen400_seed{0,1,2}.zip`. Si un modelo ya existe lo salta
(`--force` lo sobrescribe). Opciones:

- `--reward v0`: la primera recompensa, solo log-retorno
  (`saves/dqn_investor-v0_seed*.zip`).
- `--seeds 0 1 2`: qué semillas entrenar.
- `--out carpeta`: guardar en otra carpeta.

`notebooks/investor_reward.ipynb` entrena además todas las variantes de la
iteración de la recompensa (18 modelos, ~40 minutos) y las deja en `saves/`.

## Evaluar

```bash
uv run investor evaluate --period valid
```

Compara el agente final (cada semilla y su media) con comprar y mantener y con
200 agentes aleatorios, con las mismas condiciones: $100.000 iniciales en
efectivo, 10 bps por operación y ejecución al cierre. Antes de evaluar
comprueba que los pesos de los modelos son los congelados en
`configs/investor_final.json`.

`--period test` es la evaluación final sobre el período held-out. Según el
protocolo (`notebooks/investor_protocol.ipynb`) se ejecuta una sola vez y,
después, no se cambia nada del agente.

## Baselines

```bash
uv run investor baselines --period valid
```

Comprar y mantener, siempre en efectivo, siempre 50 % y el agente aleatorio
con las semillas 0 a 199 (percentiles 5, 50 y 95).

## Semillas y reproducibilidad

- **DQN:** semillas 0, 1 y 2. Se pasan a Stable-Baselines3, que fija las
  semillas de Python, NumPy y PyTorch.
- **Agente aleatorio:** semillas 0 a 199.
- **Entorno:** determinista; los precios no cambian entre episodios.
- **Versiones:** fijadas en `uv.lock`.

Para comprobar que el entrenamiento reproduce el agente congelado, se entrena
en otra carpeta y se comparan los pesos:

```bash
uv run investor train --seeds 0 --out saves/reproduccion
```

```bash
uv run investor check --dir saves/reproduccion
```

En esta máquina (CPU, Windows) el resultado es `has the frozen weights`: la
misma semilla da exactamente la misma red. En otra máquina pueden aparecer
diferencias mínimas de punto flotante (distintas CPU usan instrucciones
distintas), y en GPU el entrenamiento no es determinista. El hash se calcula
sobre los pesos y no sobre el archivo `.zip`, que cambia en cada guardado aunque
los pesos sean iguales.

## Notebooks, en orden

| Notebook | Parte | Contenido |
|---|---|---|
| [`investor_eda.ipynb`](notebooks/investor_eda.ipynb) | 7 | Análisis exploratorio |
| [`investor_reward.ipynb`](notebooks/investor_reward.ipynb) | 9 | Iteración de la recompensa (entrena los modelos) |
| [`investor_algorithm.ipynb`](notebooks/investor_algorithm.ipynb) | 10 | DQN: red, hiperparámetros, exploración, semillas |
| [`investor_baselines.ipynb`](notebooks/investor_baselines.ipynb) | 11 | Baselines en train y validación |
| [`investor_protocol.ipynb`](notebooks/investor_protocol.ipynb) | 12 | Protocolo y agente congelado |
| [`investor_metrics.ipynb`](notebooks/investor_metrics.ipynb) | 13 | Métricas y ensayo del reporte |
| [`investor_results.ipynb`](notebooks/investor_results.ipynb) | 14 | Resultados en test |

Para abrirlos: `uv run --with jupyter jupyter lab`. Se ejecutan desde la raíz
del repositorio, y los notebooks 10 a 14 necesitan los modelos de `saves/`.

## Pruebas

```bash
uv run pytest
```

Verifican que los indicadores y el entorno no usan información del futuro, la
contabilidad del entorno (costos, deriva de la posición, comprar y mantener),
la división temporal, el SHA-256 del dataset, las métricas con los ejemplos de
la actividad y que un modelo modificado no pase la verificación.

---

# RL Games (Stable-Baselines3)

The twin of [`rl_games`](https://github.com/emiliomunozai/rl_games), with the
learning algorithms taken from
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Use it as a
reference baseline: if an SB3 agent solves an env and your hand-written one
does not, the bug is in your agent.

## Setup

```bash
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 1. A single experiment: LunarLander with DQN

An experiment is a YAML file under `configs/` that sets the env, the algorithm,
the timesteps and the hyperparameters.

```bash
rlgames-sb inspect --env LunarLander-v3 --steps 1                # spaces and a sample step
rlgames-sb train   --config configs/lunarlander_dqn.yaml          # train, evaluate, save
rlgames-sb load    dqn --env LunarLander-v3 --eval                # score the saved agent
rlgames-sb render  dqn --env LunarLander-v3 --episodes 3          # watch it land
```

Training ends with a 10-episode evaluation (a mean reward of 200+ counts as
solved) and saves the model to `saves/dqn_LunarLander-v3.zip`. The run is logged
to MLflow with its hyperparameters, learning curve, final score and model:

```bash
mlflow ui          # http://127.0.0.1:5000 -> experiment "lunarlander"
```

Training resumes from an existing save, and the model keeps the hyperparameters
it was saved with. To start over, run `rlgames-sb delete dqn --env LunarLander-v3`.

## 2. Hyperparameter tuning

[Optuna](https://optuna.org/) tries hyperparameter combinations and uses the
earlier results to pick the next ones. The search is defined in the `tune:`
section of the config:

```yaml
tune:
  trials: 20
  timesteps: 100_000          # per seed
  seeds: 2                    # training runs per trial, averaged
  evals: 10                   # evaluations during each run
  eval_episodes: 5            # episodes per evaluation
  score_last: 3               # a run's score: mean of its last 3 evaluations
  search_space:
    learning_rate: {type: float, low: 1.0e-4, high: 1.0e-3, log: true}
    batch_size: {type: categorical, choices: [64, 128, 256]}
    target_update_interval: {type: categorical, choices: [250, 1000, 5000]}
    exploration_fraction: {type: float, low: 0.05, high: 0.3}
```

Write floats as `1.0e-4`, not `1e-4`, because YAML reads the latter as a string.

### Step 1: run the search

```bash
rlgames-sb tune --config configs/lunarlander_dqn.yaml
```

Each trial trains a fresh model per seed and prints one line when it ends.
The end of a run looks like this (your numbers will differ):

```
[I ...] Trial 4 finished with value: 243.25 and parameters: {...}. Best is trial 4 ...
[I ...] Trial 6 pruned.
...
Best mean reward: 251.40 (trial 16)
Best hyperparameters:

learning_rate: 0.000301
batch_size: 128
target_update_interval: 250
exploration_fraction: 0.253

Saved to configs/lunarlander_dqn_tuned.yaml. Train with them:
  rlgames-sb train --config configs/lunarlander_dqn_tuned.yaml
```

RL results vary a lot between runs, so a single run per trial lets a lucky seed
win. That is why each trial trains several seeds (the same ones for every trial)
and scores each by its last few evaluations rather than a single final one.
A trial that falls below the median of earlier trials is stopped early
(`pruned`), which saves time for the promising ones.

### Step 2: browse the results in MLflow

From the project folder:

```bash
mlflow ui          # then open http://127.0.0.1:5000
```

1. Open the experiment **lunarlander**. The tuning run is named
   `tune-dqn-20x100k-<date>-<time>`. Expand it to see one nested run per trial
   (`tune-<date>-<time>-trial-07`).
2. Sort the trials by the `eval/mean_reward` column. Pruned trials have no score
   and carry the tag `pruned = True`.
3. Select several trials and click **Compare**. The parallel coordinates plot
   shows which hyperparameter values go with high rewards.
4. Click a trial to see its learning curves, one per seed: `seed0/eval/mean_reward`,
   `seed1/rollout/ep_rew_mean`, ... Seeds that disagree a lot mean the setting is
   unstable.

The tuning run itself holds the best values (`best.*` params), the
`best_mean_reward` metric, and every tuned config it wrote, as an artifact.

### Step 3: train with the best hyperparameters

The best values are written to `configs/lunarlander_dqn_tuned.yaml`, a copy of
the config with them merged in. Tuning overwrites it each time, and every
version is also kept on its MLflow tuning run.

`train` resumes from an existing save, and a resumed model keeps its old
hyperparameters, so delete the save first:

```bash
rlgames-sb delete dqn --env LunarLander-v3
rlgames-sb train  --config configs/lunarlander_dqn_tuned.yaml
rlgames-sb load   dqn --env LunarLander-v3 --eval
rlgames-sb render dqn --env LunarLander-v3 --episodes 3
```

Trials train for 100k steps, but the full run uses the config's `timesteps`
(200k). Short trials favor settings that learn fast, so check the result. In
MLflow, compare the new `train-dqn-200k-...` run with the one from the original
config: select both and click **Compare**. If the tuned one is not clearly
better, the original hyperparameters were already good. To try again from
the tuned values, tune the tuned config:
`rlgames-sb tune --config configs/lunarlander_dqn_tuned.yaml`.

## 3. Other agents and games

Agents: `dqn`, `ppo`, `a2c`. The policy network is picked from the observation:
`MlpPolicy` for state vectors, `CnnPolicy` for images (Atari). Any Gymnasium env
id works, with or without a config (without one, SB3's defaults are used):

```bash
rlgames-sb train ppo --env CartPole-v1 --timesteps 50000
```

| Config | |
|---|---|
| `configs/lunarlander_dqn.yaml` | DQN on LunarLander |
| `configs/lunarlander_ppo.yaml` | PPO on LunarLander |
| `configs/breakout_dqn.yaml` | DQN on Atari Breakout |

**Atari.** `ALE/...` envs get the standard preprocessing (frame skip, 84x84
grayscale, 4-frame stack). They need millions of timesteps, so use a GPU
(`--device cuda`).

**Trading.** `stocks-v0` and `forex-v0` come from
[gym-anytrading](https://github.com/AminHP/gym-anytrading). The agent trains and
is evaluated on the same price history, so treat it as an RL exercise, not a
strategy.

## 4. How it works

```
configs/                # experiment YAMLs: hyperparameters and search spaces
notebooks/              # step-by-step walkthroughs of a training run
src/rl_games_sb/
├── cli.py              # argument parsing and output formatting only
├── registry.py         # which algorithms exist, where they are saved
├── envs.py             # env construction, incl. Atari and trading envs
├── config.py           # loading experiment YAMLs, sampling search spaces
├── experiments.py      # tracked training and Optuna tuning
└── tracking.py         # MLflow logging
```

`registry.py` maps agent names to SB3 classes and each (agent, env) pair to a
save file, e.g. `saves/dqn_LunarLander-v3.zip`. That file is SB3's own format
(weights, optimizer state, hyperparameters, timestep count), which is what lets
`train` resume. Using it as a library:

```python
from rl_games_sb import registry

model = registry.load_or_create("dqn", "LunarLander-v3")
model.learn(total_timesteps=50_000)
model.save(registry.save_path("dqn", "LunarLander-v3"))
```

## Commands

| Command | What it does |
|---|---|
| `version` | Show the package version |
| `list` | List agents and whether each has a save |
| `inspect` | Show an env's spaces and a few sample steps |
| `init` | Create an untrained agent and save it |
| `train` | Train an agent and save it (logged to MLflow) |
| `tune` | Search hyperparameters with Optuna (logged to MLflow) |
| `delete` | Delete a saved agent |
| `load` | Load a saved agent and show its info (`--eval` to score it) |
| `sim` | Run episodes and print them as text |
| `render` | Run episodes in a window |
