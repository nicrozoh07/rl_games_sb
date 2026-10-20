# Datos del agente inversor

El proyecto no guarda una copia de los datos: usa el dataset común del curso,
que viene dentro del paquete `gym-anytrading` y se instala con `uv sync`.

| | |
|---|---|
| Dataset | `STOCKS_GOOGL`: precios diarios de la acción de Alphabet (GOOGL) |
| Paquete | `gym-anytrading` **2.0.0**, versión fijada en `uv.lock` |
| Archivo | `gym_anytrading/datasets/data/STOCKS_GOOGL.csv` dentro del entorno (`.venv/Lib/site-packages/` en Windows, `.venv/lib/python3.11/site-packages/` en Linux/macOS) |
| SHA-256 | `1a3d459fdbcddb9362a157b07438b0fd83e6eb04406a136b5b0a10c5b0df107a` (176.301 bytes) |
| Filas | 2.335 días hábiles, del 2009-05-22 al 2018-08-29 |
| Columnas | `Date, Open, High, Low, Close, Adj Close, Volume` |

`investor.load()` lo devuelve como `timestamp, open, high, low, close, volume`.
`Adj Close` se descarta porque es igual a `Close` en todas las filas: el split
de acciones de 2014 ya viene ajustado. La prueba
`tests/test_investor.py::test_dataset_is_the_documented_one` verifica el
SHA-256, así que un cambio en los datos hace fallar las pruebas.

## División temporal

| Período | Fechas | Uso |
|---|---|---|
| train | 2009-05-22 → 2014-12-31 | entrenar al agente y ajustar la normalización |
| valid | 2015-01-01 → 2015-12-31 | comparar recompensas y elegir la penalización |
| test | 2016-01-01 → 2018-08-29 | evaluación final, una sola vez |

Las fechas están en `investor.SPLITS`.

## Exportar una copia

Para ver los datos fuera de Python:

```bash
uv run python -c "from rl_games_sb import investor; investor.load().to_csv('data/googl_daily.csv')"
```
