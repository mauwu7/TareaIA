"""
Demo de búsqueda A* sobre una cuadrícula.

- Movimientos: 4 direcciones (arriba, abajo, izquierda, derecha), costo 1 por paso.
- Heurística: distancia Manhattan  h(n) = |fila_n - fila_meta| + |col_n - col_meta|

¿Por qué es admisible?
  Para llegar a la meta hay que recorrer al menos |dfila| + |dcol| pasos, porque
  cada paso cambia solo UNA coordenada en una unidad. Los muros solo pueden
  alargar el camino real, nunca acortarlo. Por lo tanto h(n) <= costo real
  óptimo desde n: nunca sobreestima, así que A* devuelve un camino óptimo.
  (Además es consistente: h(n) <= 1 + h(n') para cualquier vecino n'.)

Comparación: la búsqueda NO informada es la misma función con h(n) = 0
(búsqueda de costo uniforme; con costos unitarios equivale a BFS).

Uso:
    python astar_demo.py                 # genera GIF + PNG
    python astar_demo.py --sin-gif       # solo PNG y tabla en consola
    python astar_demo.py --semilla 3 --tam 40 --densidad 0.3
"""
import argparse
import heapq
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ----------------------------------------------------------------------------
# Heurísticas
# ----------------------------------------------------------------------------


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def cero(a, b):
    """h = 0  ->  búsqueda no informada (costo uniforme / Dijkstra / BFS)."""
    return 0


# ----------------------------------------------------------------------------
# Búsqueda A*
# ----------------------------------------------------------------------------


def vecinos(grid, nodo):
    f, c = nodo
    for df, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nf, nc = f + df, c + dc
        if 0 <= nf < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nf, nc] == 0:
            yield (nf, nc)


def buscar(grid, inicio, meta, h):
    """
    A* genérico: expande siempre el nodo de la frontera con menor f = g + h.
    Devuelve un diccionario con el camino, su costo, el orden de expansión
    y una "foto" de la frontera después de cada expansión (para animar).
    """
    g = {inicio: 0}
    padre = {inicio: None}
    contador = 0  # desempata entradas idénticas en el heap
    # Entrada: (f, h, contador, nodo). A igual f se prefiere menor h (más cerca de la meta).
    frontera_heap = [(h(inicio, meta), h(inicio, meta), contador, inicio)]
    en_frontera = {inicio}
    cerrados = set()
    orden = []        # nodos expandidos, en orden
    fotos = []        # (nodo_expandido, frozenset(frontera))
    detalle = []      # (nodo, g, h, f) de cada expansión, para explicar

    while frontera_heap:
        f, hn, _, nodo = heapq.heappop(frontera_heap)
        if nodo in cerrados:
            continue  # entrada vieja (se encontró un camino mejor después)
        en_frontera.discard(nodo)
        cerrados.add(nodo)
        orden.append(nodo)
        detalle.append((nodo, g[nodo], hn, f))

        if nodo == meta:
            fotos.append((nodo, frozenset(en_frontera)))
            camino = []
            while nodo is not None:
                camino.append(nodo)
                nodo = padre[nodo]
            camino.reverse()
            return dict(camino=camino, costo=g[meta], orden=orden, fotos=fotos, detalle=detalle)

        for v in vecinos(grid, nodo):
            nuevo_g = g[nodo] + 1
            if v not in g or nuevo_g < g[v]:
                g[v] = nuevo_g
                padre[v] = nodo
                contador += 1
                hv = h(v, meta)
                heapq.heappush(frontera_heap, (nuevo_g + hv, hv, contador, v))
                en_frontera.add(v)
        fotos.append((nodo, frozenset(en_frontera)))

    return dict(camino=None, costo=None, orden=orden, fotos=fotos, detalle=detalle)


# ----------------------------------------------------------------------------
# Generación de la instancia
# ----------------------------------------------------------------------------


def crear_cuadricula(tam, densidad, semilla):
    """Genera un laberinto aleatorio con solución (prueba semillas hasta lograrlo)."""
    inicio, meta = (1, 1), (tam - 2, tam - 2)
    while True:
        rng = random.Random(semilla)
        grid = np.zeros((tam, tam), dtype=int)
        for f in range(tam):
            for c in range(tam):
                if rng.random() < densidad:
                    grid[f, c] = 1
        grid[inicio] = 0
        grid[meta] = 0
        if buscar(grid, inicio, meta, cero)["camino"] is not None:
            return grid, inicio, meta
        semilla += 1


# ----------------------------------------------------------------------------
# Visualización
# ----------------------------------------------------------------------------

COL_LIBRE = np.array([1.0, 1.0, 1.0])
COL_MURO = np.array([0.15, 0.15, 0.18])
COL_FRONTERA = np.array([1.0, 0.65, 0.0])
COL_CAMINO = np.array([0.85, 0.1, 0.1])
COL_INICIO = np.array([0.1, 0.7, 0.2])
COL_META = np.array([0.55, 0.15, 0.75])


def imagen(grid, inicio, meta, resultado, k, mostrar_camino):
    """Imagen RGB después de k expansiones. Los explorados se colorean por orden."""
    img = np.empty(grid.shape + (3,))
    img[:] = COL_LIBRE
    img[grid == 1] = COL_MURO
    total = len(resultado["orden"])
    cmap = matplotlib.colormaps["Blues"]
    for i, nodo in enumerate(resultado["orden"][:k]):
        img[nodo] = cmap(0.25 + 0.65 * i / max(total - 1, 1))[:3]  # claro = temprano, oscuro = tarde
    if k > 0:
        for nodo in resultado["fotos"][k - 1][1]:
            img[nodo] = COL_FRONTERA
    if mostrar_camino and resultado["camino"]:
        for nodo in resultado["camino"]:
            img[nodo] = COL_CAMINO
    img[inicio] = COL_INICIO
    img[meta] = COL_META
    return img


def preparar_ejes(ax, titulo, inicio, meta):
    ax.set_title(titulo, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(inicio[1], inicio[0], "S", ha="center", va="center", color="white", fontweight="bold")
    ax.text(meta[1], meta[0], "G", ha="center", va="center", color="white", fontweight="bold")


def animar(grid, inicio, meta, res_u, res_a, archivo, paso=3, fps=12):
    n = max(len(res_u["orden"]), len(res_a["orden"]))
    frames = list(range(0, n, paso)) + [n]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.8))
    fig.suptitle("Misma cuadrícula, mismo costo óptimo: ¿cuántos nodos hay que explorar?", fontsize=12)
    nombres = ("Búsqueda no informada (h = 0)", "A* (h = distancia Manhattan)")
    ims = []
    for ax, nombre in zip((ax1, ax2), nombres):
        preparar_ejes(ax, nombre, inicio, meta)
        ims.append(ax.imshow(imagen(grid, inicio, meta, res_u, 0, False), interpolation="nearest"))
    leyenda = ("Azul: explorado (claro = temprano)   Naranja: frontera   Rojo: camino final\n"
               "Verde: inicio (S)   Morado: meta (G)")
    fig.text(0.5, 0.01, leyenda, ha="center", fontsize=9)

    def actualizar(k):
        for im, ax, res, nombre in zip(ims, (ax1, ax2), (res_u, res_a), nombres):
            kk = min(k, len(res["orden"]))
            termino = kk == len(res["orden"])
            im.set_data(imagen(grid, inicio, meta, res, kk, termino))
            ax.set_title(f"{nombre}\nNodos expandidos: {kk}" + ("  ✔ meta alcanzada" if termino else ""), fontsize=10)
        return ims

    anim = FuncAnimation(fig, actualizar, frames=frames, blit=False)
    anim.save(archivo, writer=PillowWriter(fps=fps))
    plt.close(fig)


def figura_final(grid, inicio, meta, res_u, res_a, archivo):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    nombres = ("Búsqueda no informada (h = 0)", "A* (h = Manhattan)")
    for ax, res, nombre in zip(axes, (res_u, res_a), nombres):
        ax.imshow(imagen(grid, inicio, meta, res, len(res["orden"]), True), interpolation="nearest")
        preparar_ejes(ax, f"{nombre}\nExpandidos: {len(res['orden'])} | costo del camino: {res['costo']}", inicio, meta)
    fig.tight_layout()
    fig.savefig(archivo, dpi=130, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tam", type=int, default=30)
    p.add_argument("--densidad", type=float, default=0.25)
    p.add_argument("--semilla", type=int, default=7)
    p.add_argument("--sin-gif", action="store_true")
    args = p.parse_args()

    grid, inicio, meta = crear_cuadricula(args.tam, args.densidad, args.semilla)
    res_u = buscar(grid, inicio, meta, cero)
    res_a = buscar(grid, inicio, meta, manhattan)

    print(f"Cuadrícula {args.tam}x{args.tam}, inicio={inicio}, meta={meta}\n")
    print("Primeras expansiones de A* (nodo | g | h | f = g + h):")
    for nodo, gn, hn, fn in res_a["detalle"][:8]:
        print(f"  {nodo}  g={gn:<3} h={hn:<3} f={fn}")

    print("\nComparación:")
    print(f"  {'Algoritmo':<28}{'Costo':>7}{'Expandidos':>13}")
    print(f"  {'No informada (h=0)':<28}{res_u['costo']:>7}{len(res_u['orden']):>13}")
    print(f"  {'A* (Manhattan)':<28}{res_a['costo']:>7}{len(res_a['orden']):>13}")
    ahorro = 100 * (1 - len(res_a["orden"]) / len(res_u["orden"]))
    print(f"  A* explora {ahorro:.0f}% menos nodos y encuentra un camino igual de corto.")
    assert res_u["costo"] == res_a["costo"], "h admisible => ambos deben dar el costo óptimo"

    figura_final(grid, inicio, meta, res_u, res_a, "comparacion_astar.png")
    print("\nGuardado: comparacion_astar.png")
    if not args.sin_gif:
        animar(grid, inicio, meta, res_u, res_a, "astar_animacion.gif")
        print("Guardado: astar_animacion.gif")


if __name__ == "__main__":
    main()
