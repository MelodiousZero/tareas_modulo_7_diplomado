#  Repositorio de Tareas

Una colección centralizada de todas mis tareas académicas, proyectos y trabajos del curso.

##  Estructura del Repositorio
tareas_modulo_7_diplomado/

├── tareas/

├── tarea_1/

├── tarea_2/

└── README.md

##  Organización

Cada carpeta de curso contiene típicamente:
- **Tareas** - Ejercicios semanales o quincenales
- **Apuntes** - Materiales de estudio y documentos de referencia (si aplica)
- **Recursos** - Materiales adicionales relacionados con el curso
- **Indicaciones** - Indicaciones del docente para hacer la tarea (en formato `.pdf`)


## Setup

Este proyecto requiere **Python 3.13.9** (con esta version de python se ejecutó todo).  
Todas las dependencias están en `requirements.txt`.

### 1. Crea un ambiente virtual

| Plataforma  | Comando                     |
|-----------|-----------------------------|
| macOS / Linux | ⁠`python3.13 -m venv .ambiente_virtual` ⁠   |
| Windows | ⁠ `py -3.13 -m venv .ambiente_virtual`⁠ |

### 2. Activa tu ambiente virtual

| Plataforma  | Comando                     |
|-----------|-----------------------------|
| macOS / Linux | `source .ambiente_virtual/bin/activate`   |
| Windows (cmd) | `.ambiente_virtual\Scripts\activate.bat` |
| Windows (PowerShell) | `.ambiente_virtual\Scripts\Activate.ps1` |

### 3. Instala las depedencias

```bash
pip install -r requirements.txt
```

Listo, procura usar este ambiente virtual para ejecutar los `.ipynb` o los `.py`.
