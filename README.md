# Demo de búsqueda A*

Demostración de A* y búsqueda no informada sobre una cuadrícula. El programa
genera `comparacion_astar.png` y, salvo que se indique lo contrario,
`astar_animacion.gif` en esta carpeta.

## Requisitos

- Python 3.9 o posterior
- `matplotlib`, `numpy` y `Pillow` (se instalan desde `requirements.txt`)

## Preparar el entorno en Windows PowerShell

Desde esta carpeta, crea y activa el entorno virtual e instala las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell impide activar el entorno por la política de ejecución, puedes
invocar directamente el Python del entorno:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe astar_demo.py
```

## Ejecutar

Con el entorno activado:

```powershell
python astar_demo.py
```

Opciones disponibles:

```powershell
python astar_demo.py --sin-gif
python astar_demo.py --semilla 3 --tam 40 --densidad 0.3
```
