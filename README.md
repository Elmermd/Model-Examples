# Model Examples

Educational examples demonstrating machine learning concepts with clear, step-by-step explanations.

##  Examples

### Decision Trees

- **[decision_tree_samples_and_leaves.py](examples/decision_tree_samples_and_leaves.py)** *(English)*: In-depth explanation of how decision trees work
  - Understand samples vs features
  - See how nodes split data
  - Trace individual predictions through the tree
  - Visualize sample distribution in leaves

- **[arbol_decision_prediccion_casas.py](examples/arbol_decision_prediccion_casas.py)** *(Español)*: Ejemplo completo de árbol de decisión para predicción de precios de casas
  - Dataset de 20 casas con visualizaciones
  - División train/test y evaluación de métricas
  - Análisis de overfitting e importancia de características
  - Visualización del árbol y residuos

##  Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Examples

```bash
# Decision tree - Samples and leaves (English)
python examples/decision_tree_samples_and_leaves.py

# Árbol de decisión - Predicción de casas (Español)
python examples/arbol_decision_prediccion_casas.py
```

##  What You'll Learn

- **Samples vs Features**: Clear distinction between observations (rows) and attributes (columns)
- **Tree Structure**: How nodes make decisions and route samples to leaves
- **Predictions**: How the tree averages values in leaves to make predictions
- **Visualization**: Reading sklearn's decision tree plots

##  Who Is This For?

- Beginners learning machine learning
- Students studying decision trees
- Anyone who wants to understand ML models at a deeper level

##  License

MIT License - Feel free to use for learning and teaching!
