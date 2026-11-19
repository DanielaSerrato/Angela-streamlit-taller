# Cinco datos para tus gráficas

Cinco archivos, cada uno con una pregunta y una gráfica que le queda bien.
Los datos son inventados: puedes equivocarte sin consecuencias.

| Archivo | Filas | Columnas | La gráfica que le queda |
|---|---|---|---|
| `01_ventas_mensuales.csv` | 24 | fecha, mes, ventas_millones, publicidad_millones | línea, dos series |
| `02_clientes_por_ciudad_canal.csv` | 20 | ciudad, canal, clientes | barras ordenadas y mapa de calor |
| `03_clientes.csv` | 180 | cliente, segmento, compras, gasto_medio, edad | dispersión, caja, correlación |
| `04_entregas.csv` | 240 | tipo, tiempo_horas, distancia_km | histograma, densidad, caja |
| `05_productos.csv` | 12 | producto, categoria, precio, unidades, margen_pct | barras horizontales, burbuja |

## Cómo se cargan

```r
datos <- read.csv("03_clientes.csv")           # R
```

```python
datos = pd.read_csv("03_clientes.csv")         # Python
```

## El reto, con el que elijas

1. Una gráfica que responda la pregunta del archivo.
2. Un título que diga la afirmación, no el tema.
3. Una frase de límite: qué no se puede concluir.

El detalle de cada uno (pregunta, pista y trampa) está en `..\datos_practica.pdf`.
