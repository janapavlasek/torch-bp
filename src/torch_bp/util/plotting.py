import matplotlib.colors as cm
from torch_bp.graph.mrf_graph import MRFGraph
from torch_bp.graph.factor_graph import FactorGraph

COLORS = ["tab:orange",
          "tab:pink",
          "tab:cyan",
          "tab:green",
          "tab:purple",
          "tab:blue",
          "tab:red",
          "tab:olive",
          "tab:brown",
          "tab:grey"]

CMAPS = [cm.LinearSegmentedColormap.from_list("map", [cm.to_rgba(c, alpha=0.), c]) for c in COLORS]


def plot_graph(ax, nodes, g, c='tab:red'):
    if isinstance(g, MRFGraph):
        for e in g.edges:
            n1, n2 = e
            ax.plot(nodes[[n1, n2], 0], nodes[[n1, n2], 1], c=c)
    elif isinstance(g, FactorGraph):
        for factor_cluster in g.factor_clusters.values():
            nbrs = factor_cluster.neighbours
            n = len(nbrs)
            [ax.plot(nodes[[nbrs[i], nbrs[j]], 0], nodes[[nbrs[i], nbrs[j]], 1], c=c)
             for i in range(n-1) for j in range(i+1, n)]

    ax.scatter(nodes[:, 0], nodes[:, 1], marker='x', c=c, zorder=2)


def plot_dists(ax, dists, lims, cmap=None):
    if cmap is None:
        cmap = CMAPS
    if not isinstance(cmap, list):
        cmap = [cmap for _ in dists]
    for p, c in zip(dists, cmap):
        X, Y, Z = p.eval_grid(lims)
        ax.contour(X, Y, Z, cmap=c)


def plot_particles(ax, particles, node_pos, graph=None, dists=None, weights=None, mle=None,
                   lims=None, cmap=None, colors=None, min_alpha=0.1):
    # Use default colour maps.
    if colors is None:
        colors = COLORS
    if cmap is None:
        cmap = CMAPS

    if dists is not None:
        plot_dists(ax, dists, lims, cmap=cmap)
    if graph is not None:
        plot_graph(ax, node_pos, graph)
    if weights is not None:
        weights = weights / weights.max(-1, keepdims=True)[0]
        # Put a minimum value on the weights so the particle doesn't fully disappear.
        weights = weights.clip(min_alpha, 1)

    for i in range(particles.shape[0]):
        pts = particles[i, :, :].cpu().numpy()
        alpha = weights[i, :] if weights is not None else 0.5
        ax.scatter(pts[:, 0], pts[:, 1], zorder=2, c=colors[i], alpha=alpha)

    if mle is not None:
        for i, est in enumerate(mle):
            ax.scatter(est[0], est[1], c=colors[i], marker="X",
                       zorder=3, s=100, edgecolors="white")

    if lims is not None:
        ax.set_xlim(*lims[:2])
        ax.set_ylim(*lims[2:])
