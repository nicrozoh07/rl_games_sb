# Agente inversor con DQN (proyecto final de Aprendizaje por Refuerzo)

Un agente de Reinforcement Learning decide cada día qué fracción de un capital
de $100.000 invertir en la acción de Alphabet (GOOGL): 0 %, 50 % o 100 %. Paga
10 bps por cada operación y se compara con comprar y mantener y con un agente
aleatorio en un período que no vio al entrenar.

Este repositorio es un fork de
[`emiliomunozai/rl_games_sb`](https://github.com/emiliomunozai/rl_games_sb): el
proyecto reutiliza su DQN (Stable-Baselines3), sus configuraciones y su
seguimiento con MLflow. La documentación original del curso, traducida al español,
sigue más abajo.

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

Documentación del repositorio original del curso, traducida al español.

Es el gemelo de [`rl_games`](https://github.com/emiliomunozai/rl_games), con
los algoritmos de aprendizaje tomados de
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Sirve como
referencia: si un agente de SB3 resuelve un entorno y el que programaste a mano
no, el error está en tu agente.

## Instalación

```bash
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 1. Un experimento: LunarLander con DQN

Un experimento es un archivo YAML en `configs/` que define el entorno, el
algoritmo, el número de pasos y los hiperparámetros.

```bash
rlgames-sb inspect --env LunarLander-v3 --steps 1                # espacios y un paso de ejemplo
rlgames-sb train   --config configs/lunarlander_dqn.yaml          # entrenar, evaluar y guardar
rlgames-sb load    dqn --env LunarLander-v3 --eval                # evaluar el agente guardado
rlgames-sb render  dqn --env LunarLander-v3 --episodes 3          # verlo aterrizar
```

El entrenamiento termina con una evaluación de 10 episodios (una recompensa
media de 200 o más se considera resuelto) y guarda el modelo en
`saves/dqn_LunarLander-v3.zip`. La corrida queda registrada en MLflow con sus
hiperparámetros, su curva de aprendizaje, su puntaje final y el modelo:

```bash
mlflow ui          # http://127.0.0.1:5000 -> experimento "lunarlander"
```

El entrenamiento continúa desde un modelo guardado si ya existe, y el modelo
conserva los hiperparámetros con los que se guardó. Para empezar de cero,
ejecuta `rlgames-sb delete dqn --env LunarLander-v3`.

## 2. Ajuste de hiperparámetros

[Optuna](https://optuna.org/) prueba combinaciones de hiperparámetros y usa los
resultados anteriores para elegir las siguientes. La búsqueda se define en la
sección `tune:` de la configuración:

```yaml
tune:
  trials: 20
  timesteps: 100_000          # por semilla
  seeds: 2                    # entrenamientos por prueba, promediados
  evals: 10                   # evaluaciones durante cada entrenamiento
  eval_episodes: 5            # episodios por evaluación
  score_last: 3               # puntaje de un entrenamiento: media de sus 3 últimas evaluaciones
  search_space:
    learning_rate: {type: float, low: 1.0e-4, high: 1.0e-3, log: true}
    batch_size: {type: categorical, choices: [64, 128, 256]}
    target_update_interval: {type: categorical, choices: [250, 1000, 5000]}
    exploration_fraction: {type: float, low: 0.05, high: 0.3}
```

Escribe los decimales como `1.0e-4` y no como `1e-4`, porque YAML lee lo
segundo como texto.

### Paso 1: ejecutar la búsqueda

```bash
rlgames-sb tune --config configs/lunarlander_dqn.yaml
```

Cada prueba entrena un modelo nuevo por semilla e imprime una línea al terminar.
El final de una búsqueda se ve así (tus números serán distintos):

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

Los resultados de RL cambian mucho de una corrida a otra, así que con un solo
entrenamiento por prueba podría ganar una semilla con suerte. Por eso cada
prueba entrena varias semillas (las mismas en todas las pruebas) y califica cada
una con sus últimas evaluaciones, no con una sola evaluación final. Una prueba
que queda por debajo de la mediana de las anteriores se detiene antes de tiempo
(`pruned`), lo que ahorra tiempo para las prometedoras.

### Paso 2: revisar los resultados en MLflow

Desde la carpeta del proyecto:

```bash
mlflow ui          # luego abre http://127.0.0.1:5000
```

1. Abre el experimento **lunarlander**. La búsqueda se llama
   `tune-dqn-20x100k-<fecha>-<hora>`. Despliégala para ver una corrida anidada
   por prueba (`tune-<fecha>-<hora>-trial-07`).
2. Ordena las pruebas por la columna `eval/mean_reward`. Las pruebas detenidas
   no tienen puntaje y llevan la etiqueta `pruned = True`.
3. Selecciona varias pruebas y haz clic en **Compare**. El gráfico de
   coordenadas paralelas muestra qué valores de los hiperparámetros van con
   recompensas altas.
4. Haz clic en una prueba para ver sus curvas de aprendizaje, una por semilla:
   `seed0/eval/mean_reward`, `seed1/rollout/ep_rew_mean`, ... Si las semillas
   difieren mucho, esa configuración es inestable.

La corrida de la búsqueda guarda los mejores valores (parámetros `best.*`), la
métrica `best_mean_reward` y, como artefacto, cada configuración ajustada que
escribió.

### Paso 3: entrenar con los mejores hiperparámetros

Los mejores valores se escriben en `configs/lunarlander_dqn_tuned.yaml`, una
copia de la configuración con esos valores incorporados. Cada búsqueda lo
sobrescribe, y cada versión también queda guardada en su corrida de MLflow.

`train` continúa desde un modelo guardado, y un modelo que continúa conserva
sus hiperparámetros anteriores, así que primero hay que borrar el modelo:

```bash
rlgames-sb delete dqn --env LunarLander-v3
rlgames-sb train  --config configs/lunarlander_dqn_tuned.yaml
rlgames-sb load   dqn --env LunarLander-v3 --eval
rlgames-sb render dqn --env LunarLander-v3 --episodes 3
```

Las pruebas entrenan 100k pasos, pero el entrenamiento completo usa los
`timesteps` de la configuración (200k). Las pruebas cortas favorecen las
configuraciones que aprenden rápido, así que conviene comprobar el resultado.
En MLflow, compara la nueva corrida `train-dqn-200k-...` con la de la
configuración original: selecciona ambas y haz clic en **Compare**. Si la
ajustada no es claramente mejor, los hiperparámetros originales ya eran buenos.
Para volver a intentarlo desde los valores ajustados, ajusta la configuración
ajustada: `rlgames-sb tune --config configs/lunarlander_dqn_tuned.yaml`.

## 3. Otros agentes y juegos

Agentes: `dqn`, `ppo`, `a2c`. La red de la política se elige según la
observación: `MlpPolicy` para vectores de estado y `CnnPolicy` para imágenes
(Atari). Sirve cualquier identificador de entorno de Gymnasium, con o sin
configuración (sin ella se usan los valores por defecto de SB3):

```bash
rlgames-sb train ppo --env CartPole-v1 --timesteps 50000
```

| Configuración | |
|---|---|
| `configs/lunarlander_dqn.yaml` | DQN en LunarLander |
| `configs/lunarlander_ppo.yaml` | PPO en LunarLander |
| `configs/breakout_dqn.yaml` | DQN en Atari Breakout |

**Atari.** Los entornos `ALE/...` reciben el preprocesamiento estándar (salto de
4 cuadros, escala de grises de 84x84 y pila de 4 cuadros). Necesitan millones
de pasos, así que conviene usar una GPU (`--device cuda`).

**Trading.** `stocks-v0` y `forex-v0` vienen de
[gym-anytrading](https://github.com/AminHP/gym-anytrading). El agente se
entrena y se evalúa sobre la misma serie de precios, así que hay que tomarlo
como un ejercicio de RL y no como una estrategia.

## 4. Cómo funciona

```
configs/                # YAML de los experimentos: hiperparámetros y espacios de búsqueda
notebooks/              # recorridos paso a paso de un entrenamiento
src/rl_games_sb/
├── cli.py              # solo lectura de argumentos y formato de la salida
├── registry.py         # qué algoritmos existen y dónde se guardan
├── envs.py             # construcción de entornos, incluidos Atari y trading
├── config.py           # lectura de los YAML y muestreo de los espacios de búsqueda
├── experiments.py      # entrenamiento registrado y ajuste con Optuna
└── tracking.py         # registro en MLflow
```

`registry.py` asocia cada nombre de agente con su clase de SB3 y cada par
(agente, entorno) con un archivo guardado, por ejemplo
`saves/dqn_LunarLander-v3.zip`. Ese archivo está en el formato propio de SB3
(pesos, estado del optimizador, hiperparámetros y número de pasos), y eso es lo
que permite que `train` continúe un entrenamiento. Para usarlo como librería:

```python
from rl_games_sb import registry

model = registry.load_or_create("dqn", "LunarLander-v3")
model.learn(total_timesteps=50_000)
model.save(registry.save_path("dqn", "LunarLander-v3"))
```

## Comandos

| Comando | Qué hace |
|---|---|
| `version` | Muestra la versión del paquete |
| `list` | Lista los agentes e indica si cada uno tiene un modelo guardado |
| `inspect` | Muestra los espacios de un entorno y algunos pasos de ejemplo |
| `init` | Crea un agente sin entrenar y lo guarda |
| `train` | Entrena un agente y lo guarda (registrado en MLflow) |
| `tune` | Busca hiperparámetros con Optuna (registrado en MLflow) |
| `delete` | Borra un agente guardado |
| `load` | Carga un agente guardado y muestra su información (`--eval` para evaluarlo) |
| `sim` | Ejecuta episodios y los imprime como texto |
| `render` | Ejecuta episodios en una ventana |
