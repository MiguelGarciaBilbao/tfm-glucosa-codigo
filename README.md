# Predicción personalizada de glucosa e hipoglucemias en diabetes tipo 1 (OhioT1DM)

Código del Trabajo Fin de Máster de Miguel García Bilbao. Predice la glucosa intersticial (CGM)
a 30 y 60 minutos sobre el dataset OhioT1DM con un modelo global probabilístico (media y
varianza de cada horizonte) y evalúa la detección temprana de hipoglucemias con una
alarma `P(y < 70 mg/dL) > α`, además de las métricas de regresión.

## Requisitos e instalación

Python 3.11 o superior (desarrollado con 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Datos

OhioT1DM **no se incluye**: se distribuye bajo un *Data Use Agreement* (DUA) firmado entre
instituciones con sus autores (Marling y Bunescu, Ohio University). Con el acceso concedido,
los XML se colocan así:

```
datasets/OhioT1DM/{2018,2020}/{train,test}/<id>-ws-{training,testing}.xml
```

Después se generan los derivados:

```bash
python -m tfm.data.build            # XML -> data/interim/ (parquet por paciente y split)
python -m tfm.data.build_processed  # -> data/processed/ (ventanas .npz)
```

### Validación externa: HUPA-UCM

HUPA-UCM (Hidalgo et al., 2024) tampoco se incluye. Se descarga de Mendeley Data
(doi:10.17632/3hbcscwz44.1, licencia CC BY-NC-ND 4.0) y la carpeta descomprimida se coloca así:

```
datasets/external/HUPA-UCM/{Preprocessed,Raw_Data}/
```

```bash
python -m tfm.data.hupa             # Libre crudo + tratamiento -> data/external/hupa/
```

La glucosa se lee de las exportaciones crudas del FreeStyle Libre y no de `Preprocessed/`, que
interpola todos los huecos. Las decisiones de armonización (unidades, pacientes excluidos,
registro del tratamiento) están documentadas en `src/tfm/data/hupa.py`.
