import numpy as np


class MultiRobot(object):
    def __init__(self, num_robots, lims=[0, 10], pos=None, radius=1, dim=2, full_conn=True, min_dist=0.4):
        self.N = num_robots
        self.radius = radius

        if pos is not None:
            self.pos = np.array(pos)
        else:
            self.create_graph_walk(lims, dim, full_conn)

    def states(self):
        return self.pos.copy()

    def create_graph(self, lims, dim=2, full_conn=True):
        self.pos = np.random.uniform(*lims[:dim], (self.N, dim))

        graph = self.calc_adjacency(self.pos)

        if full_conn:
            # Ensure the graph is fully connected.
            lost_robots = self._calc_lost_robots(graph)
            while len(lost_robots) > 0:
                self.pos[lost_robots, :] = np.random.uniform(*lims[:dim], size=(len(lost_robots), dim))
                graph = self.calc_adjacency(self.pos)
                lost_robots = self._calc_lost_robots(graph)

        # Grab edges.
        self.edges = np.stack(np.triu(graph).nonzero(), axis=1)

        # Grab edge distances.
        dists = np.linalg.norm(self.pos[..., None, :] - self.pos, axis=-1)
        self.edge_lens = dists[self.edges[:, 0], self.edges[:, 1]]

    def _draw_new_robot(self, pos, lims, noise=0.5, rad=3):
        closest = np.random.randint(0, pos.shape[0])
        angle = np.random.uniform(0, np.pi)
        new_pos = pos[closest, :] + rad * np.array([np.cos(angle), np.sin(angle)])
        # Add noise.
        new_pos += np.random.normal(0, noise)

        while np.any(new_pos <= lims[0] * np.ones(2)) or np.any(new_pos > lims[1] * np.ones(2)):
            # If out of the limts, draw again.
            closest = np.random.randint(0, pos.shape[0])
            angle = np.random.uniform(0, np.pi)
            new_pos = pos[closest, :] + rad * np.array([np.cos(angle), np.cos(angle)])
            # Add noise.
            new_pos += np.random.normal(0, noise)

        return new_pos

    def _connect_robots(self, pos, lims):
        graph = self.calc_adjacency(pos)
        lost_robots = self._calc_lost_robots(graph)
        while len(lost_robots) > 0:
            pos = np.delete(pos, lost_robots[-1], axis=0)
            new_pos = self._draw_new_robot(pos, lims)
            pos = np.concatenate([pos, np.expand_dims(new_pos, 0)])

            graph = self.calc_adjacency(pos)
            lost_robots = self._calc_lost_robots(graph)

        return pos

    def _push_apart(self, pos, lims, min_dist):
        graph = self.calc_adjacency(pos)
        edges = np.stack(np.triu(graph).nonzero(), axis=1)
        dists = np.linalg.norm(pos[..., None, :] - pos, axis=-1)
        edge_lens = dists[edges[:, 0], edges[:, 1]]

        too_small, = np.nonzero(edge_lens < min_dist)
        while len(too_small) > 0:
            pos = np.delete(pos, edges[too_small[0], 0], axis=0)
            new_pos = self._draw_new_robot(pos, lims)
            pos = np.concatenate([pos, np.expand_dims(new_pos, 0)])

            pos = self._connect_robots(pos, lims)

            graph = self.calc_adjacency(pos)
            edges = np.stack(np.triu(graph).nonzero(), axis=1)
            dists = np.linalg.norm(pos[..., None, :] - pos, axis=-1)
            edge_lens = dists[edges[:, 0], edges[:, 1]]
            too_small, = np.nonzero(edge_lens < min_dist)

        return pos

    def create_graph_walk(self, lims, dim=2, full_conn=True, min_dist=1, max_nbrs=4):
        pos = np.random.uniform(*lims[:dim], (1, dim))

        while pos.shape[0] < self.N:
            new_pos = self._draw_new_robot(pos, lims)
            pos = np.concatenate([pos, np.expand_dims(new_pos, 0)])

            pos = self._connect_robots(pos, lims)

            # Check if any are too close.
            if min_dist is not None:
                pos = self._push_apart(pos, lims, min_dist)

            num_nbrs = self.calc_adjacency(pos).sum(-1)
            too_conn, = np.nonzero(num_nbrs > max_nbrs)
            while len(too_conn) > 0:
                pos = np.delete(pos, too_conn, axis=0)
                num_nbrs = self.calc_adjacency(pos).sum(-1)
                too_conn, = np.nonzero(num_nbrs > max_nbrs)

        self.pos = pos
        # Grab edges.
        graph = self.calc_adjacency(pos)
        self.edges = np.stack(np.triu(graph).nonzero(), axis=1)

        # Grab edge distances.
        dists = np.linalg.norm(self.pos[..., None, :] - self.pos, axis=-1)
        self.edge_lens = dists[self.edges[:, 0], self.edges[:, 1]]

    def calc_adjacency(self, pos):
        delta = pos[..., None, :] - pos
        dists = np.linalg.norm(delta, axis=-1)
        adj = dists <= self.radius
        np.fill_diagonal(adj, 0)
        return adj

    def _calc_lost_robots(self, adj_mat):
        conn = adj_mat.sum(axis=1)
        return np.nonzero(conn < 1)[0]
