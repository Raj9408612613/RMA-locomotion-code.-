"""Terrain bank for the Isaac Lab port of rsg_a1_task.

RaiSim gives every environment its own world, so the original code removes and re-adds a
height map per environment on reset. Isaac Lab shares one stage between all environments,
so instead we pre-generate a grid of tiles (one column per terrain kind / difficulty level,
one row per random variant) and "change terrain" by moving the robot to another tile.

The heights are kept as a torch tensor so `heights_at` plays the role of `hm_->getHeight`.
This module only depends on numpy / torch so it can be tested without Isaac Sim.
"""

import numpy as np
import torch

KIND_ROUGH, KIND_UP, KIND_DOWN = 0, 1, 2

# Same lists as Environment.hpp
STEP_HEIGHT_LIST = [0.1, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23]
NUM_DOWN_LEVELS = len(STEP_HEIGHT_LIST) - 3


class RaisimPerlin:
    """raisim::PerlinNoise (raisimLib include/raisim/Terrain.hpp): Ken Perlin's improved noise with a
    seeded permutation table, sampled at z = 0 as raisim::TerrainGenerator does."""

    def __init__(self, rng):
        p = rng.permutation(256)
        self.p = np.concatenate([p, p])

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _grad(h, x, y, z):
        h = h & 15
        u = np.where(h < 8, x, y)
        v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
        return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)

    def sample(self, x, y):
        p, z = self.p, 0.0
        X, Y = np.floor(x).astype(int) & 255, np.floor(y).astype(int) & 255
        x, y = x - np.floor(x), y - np.floor(y)
        u, v, w = self._fade(x), self._fade(y), self._fade(z)
        A, B = p[X] + Y, p[X + 1] + Y
        AA, AB, BA, BB = p[A], p[A + 1], p[B], p[B + 1]
        g = self._grad
        lerp = lambda t, a, b: a + t * (b - a)
        return lerp(w, lerp(v, lerp(u, g(p[AA], x, y, z), g(p[BA], x - 1, y, z)),
                            lerp(u, g(p[AB], x, y - 1, z), g(p[BB], x - 1, y - 1, z))),
                    lerp(v, lerp(u, g(p[AA + 1], x, y, z - 1), g(p[BA + 1], x - 1, y, z - 1)),
                         lerp(u, g(p[AB + 1], x, y - 1, z - 1), g(p[BB + 1], x - 1, y - 1, z - 1))))


def fractal(n, hs, frequency, z_scale, rng, octaves=2, lacunarity=2.0, gain=0.25,
            map_size=12.0, map_samples=200):
    """raisim::TerrainGenerator::generatePerlinFractalTerrain for a map of `map_samples` points over
    `map_size` metres (as in Environment.hpp), resampled onto an n x n tile with spacing `hs`.
      - noise coordinate of sample i: frequency * map_size / map_samples * i   (frequency is per metre)
      - height = zScale * (sum_o gain^o * perlin(lacunarity^o * coord) * fractalBound * 0.5 + 0.5)
        with fractalBound = 1 / sum_o gain^o, i.e. heights lie in [0, zScale]
      - a RaiSim height map interpolates linearly between samples spaced map_size / (map_samples - 1)."""
    perlin = RaisimPerlin(rng)
    spacing = map_size / (map_samples - 1)
    m = int(np.ceil((n - 1) * hs / spacing)) + 2
    idx = np.arange(m) * (frequency * map_size / map_samples)
    X, Y = np.meshgrid(idx, idx, indexing="ij")
    total, amp, bound = perlin.sample(X, Y), 1.0, 1.0
    for _ in range(1, octaves):
        X, Y = X * lacunarity, Y * lacunarity
        amp *= gain
        bound += amp
        total = total + perlin.sample(X, Y) * amp
    grid = z_scale * (total / bound * 0.5 + 0.5)
    # bilinear resample of the sample grid onto the tile grid
    q = np.arange(n) * hs / spacing
    i0 = np.minimum(np.floor(q).astype(int), m - 2)
    f = q - i0
    rows = grid[i0] * (1 - f)[:, None] + grid[i0 + 1] * f[:, None]
    return rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]


def rough_tile(n, hs, rng):
    # add_random_terrain: frequency 10, zScale 0.22, octaves 2, lacunarity 2, gain 0.25, 200 samples / 12 m
    return fractal(n, hs, 10.0, 0.22, rng)


def isaaclab_stairs_tile(n, hs, step_height, inverted, step_width=0.3, platform_width=3.0, border_width=1.0):
    """Height field of Isaac Lab's MeshPyramidStairsTerrain / MeshInvertedPyramidStairsTerrain
    (isaaclab/terrains/trimesh/mesh_terrains.py, v2.3.2) with ROUGH_TERRAINS_CFG's stair settings:
    1 m flat border at 0, square rings of `step_width` wide steps, ring k at +-(k+1)*h, centre platform at
    +-(num_steps+1)*h. Pyramid (centre on top) = walking out goes DOWN; inverted (pit) = walking out goes UP."""
    size = n * hs
    num_steps = int((size - 2 * border_width - platform_width) // (2 * step_width) + 1)
    sign = -1.0 if inverted else 1.0
    c = np.arange(n) * hs
    d = np.minimum(np.minimum(c, size - c)[:, None], np.minimum(c, size - c)[None, :])  # distance to tile edge
    k = np.floor((d - border_width) / step_width + 1e-9)
    h = np.where(d < border_width, 0.0, np.where(k >= num_steps, num_steps + 1, k + 1) * step_height)
    return sign * h


class TerrainBank:
    def __init__(self, tile_size=8.0, horizontal_scale=0.05, vertical_scale=0.005, num_variants=5,
                 border=3.0, seed=0, device="cpu"):
        rng = np.random.default_rng(seed)
        self.hs, self.vs = horizontal_scale, vertical_scale
        n = int(round(tile_size / horizontal_scale))
        b = int(round(border / horizontal_scale))
        self.tile_size, self.tile_n, self.num_variants = n * horizontal_scale, n, num_variants

        # columns: rough | up levels 0..6 | down levels 0..3
        self.columns = [(KIND_ROUGH, 0)]
        self.columns += [(KIND_UP, l) for l in range(len(STEP_HEIGHT_LIST))]
        self.columns += [(KIND_DOWN, l) for l in range(NUM_DOWN_LEVELS)]
        n_cols = len(self.columns)

        H = np.zeros((n_cols * n + 2 * b, num_variants * n + 2 * b))
        for c, (kind, level) in enumerate(self.columns):
            for r in range(num_variants):
                if kind == KIND_ROUGH:
                    tile = rough_tile(n, horizontal_scale, rng)
                else:
                    # UP: start at the bottom of an inverted pyramid; DOWN: start on top of a pyramid
                    tile = isaaclab_stairs_tile(n, horizontal_scale, STEP_HEIGHT_LIST[level],
                                                inverted=(kind == KIND_UP))
                H[b + c * n:b + (c + 1) * n, b + r * n:b + (r + 1) * n] = tile
        self.height_field = np.round(H / vertical_scale).astype(np.int16)
        self.origin = np.array([-(H.shape[0] - 1) * horizontal_scale / 2, -(H.shape[1] - 1) * horizontal_scale / 2])

        self.device = device
        self.heights = torch.tensor(self.height_field * vertical_scale, dtype=torch.float32, device=device)
        self._origin_t = torch.tensor(self.origin, dtype=torch.float32, device=device)

        # column lookup table [kind, level] -> column, and tile corner world coordinates
        col_table = torch.full((3, len(STEP_HEIGHT_LIST)), -1, dtype=torch.long)
        for c, (kind, level) in enumerate(self.columns):
            col_table[kind, level] = c
        self.col_table = col_table.to(device)
        cc, rr = torch.meshgrid(torch.arange(n_cols), torch.arange(num_variants), indexing="ij")
        corners = torch.stack([(b + cc * n) * horizontal_scale, (b + rr * n) * horizontal_scale], -1)
        self.tile_corner = corners.to(device=device, dtype=torch.float32) + self._origin_t
        # robot spawn offset inside a tile: rough about 2 m from the edge (as RaiSim),
        # stairs on the centre platform of the Isaac Lab pyramid, facing +x
        self.spawn_local = torch.tensor([[2.0, self.tile_size / 2],
                                         [self.tile_size / 2, self.tile_size / 2],
                                         [self.tile_size / 2, self.tile_size / 2]], device=device)
        self.n_border = b

    def spawn_xy(self, kind, level, variant):
        col = self.col_table[kind, level]
        return self.tile_corner[col, variant] + self.spawn_local[kind]

    def heights_at(self, xy):
        """Bilinear height lookup at world xy (..., 2), clamped to the terrain bounds."""
        g = (xy - self._origin_t) / self.hs
        nx, ny = self.heights.shape
        gx = g[..., 0].clamp(0, nx - 1.0001)
        gy = g[..., 1].clamp(0, ny - 1.0001)
        x0, y0 = gx.floor().long(), gy.floor().long()
        fx, fy = gx - x0, gy - y0
        h = self.heights
        return ((h[x0, y0] * (1 - fx) + h[x0 + 1, y0] * fx) * (1 - fy)
                + (h[x0, y0 + 1] * (1 - fx) + h[x0 + 1, y0 + 1] * fx) * fy)

    def mesh_parts(self):
        """(name, first row, last row + 1, slope_threshold) blocks for Isaac Lab's height-field-to-mesh:
        RaiSim rough terrain stays a plain triangulation, Isaac Lab stairs get vertical risers
        (slope_threshold 0.75, as in ROUGH_TERRAINS_CFG). Blocks share one boundary row."""
        split = self.n_border + self.tile_n  # rough column ends here
        return [("rough", 0, split + 1, None), ("stairs", split, self.height_field.shape[0], 0.75)]

    def mesh(self):
        """Vertices / triangles of the height field (two triangles per cell) in world frame."""
        nx, ny = self.height_field.shape
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        verts = np.stack([self.origin[0] + ix * self.hs, self.origin[1] + iy * self.hs,
                          self.height_field * self.vs], -1).reshape(-1, 3).astype(np.float32)
        idx = (ix * ny + iy)[:-1, :-1].reshape(-1)
        a, b_, c, d = idx, idx + ny, idx + 1, idx + ny + 1
        tris = np.concatenate([np.stack([a, b_, d], -1), np.stack([a, d, c], -1)]).astype(np.uint32)
        return verts, tris
