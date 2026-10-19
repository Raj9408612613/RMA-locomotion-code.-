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
STEP_LENGTH = 0.6  # step_length_list[rand() % max_step_idx] with max_step_idx = 1
NUM_DOWN_LEVELS = len(STEP_HEIGHT_LIST) - 3


def perlin(xs, ys, rng):
    """2D gradient noise evaluated at continuous lattice coordinates xs (nx,), ys (ny,)."""
    gx, gy = int(np.floor(xs.max())) + 2, int(np.floor(ys.max())) + 2
    angles = rng.uniform(0, 2 * np.pi, (gx, gy))
    grads = np.stack([np.cos(angles), np.sin(angles)], -1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x0, y0 = np.floor(X).astype(int), np.floor(Y).astype(int)
    fx, fy = X - x0, Y - y0

    def dot(ix, iy, dx, dy):
        g = grads[ix, iy]
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = dot(x0, y0, fx, fy)
    n10 = dot(x0 + 1, y0, fx - 1, fy)
    n01 = dot(x0, y0 + 1, fx, fy - 1)
    n11 = dot(x0 + 1, y0 + 1, fx - 1, fy - 1)
    u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v


def fractal(n, hs, frequency, z_scale, rng, octaves=2, lacunarity=2.0, gain=0.25, ref_size=12.0):
    """Fractal noise tile of n x n samples. `frequency` is periods per `ref_size` metres
    (RaiSim TerrainProperties used 12 m maps). Amplitude semantics of RaiSim's generator are
    not documented; here z_scale multiplies Perlin noise whose range is roughly [-0.7, 0.7]."""
    coords = np.arange(n) * hs
    out = np.zeros((n, n))
    amp, freq = 1.0, frequency / ref_size
    for _ in range(octaves):
        out += amp * perlin(coords * freq, coords * freq, rng)
        amp *= gain
        freq *= lacunarity
    return z_scale * out


def rough_tile(n, hs, rng):
    # add_random_terrain: frequency 10, zScale 0.22, octaves 2, lacunarity 2, gain 0.25
    return fractal(n, hs, 10.0, 0.22, rng)


def stairs_tile(n, hs, step_height, rng, noise_z_scale):
    """add_stairs: 1.2 m start platform at 0.2 m, 0.6 m steps along +x, a 1.2 m landing
    from 4.8 m to 6.0 m, then steps continue. Half of the steps get a 2.5 cm chamfer on
    their first row. Down stairs (step_height < 0) get fractal noise added."""
    xs = np.arange(n) * hs
    h = np.full(n, 0.2)
    plat_end, land0, land1 = 1.2, 4.8, 6.0
    in_first = (xs >= plat_end) & (xs < land0)
    k1 = np.floor((xs - plat_end) / STEP_LENGTH)
    h[in_first] = 0.2 + (k1[in_first] + 1) * step_height
    n_first = int(round((land0 - plat_end) / STEP_LENGTH))
    top = 0.2 + n_first * step_height
    h[(xs >= land0) & (xs < land1)] = top
    in_second = xs >= land1
    k2 = np.floor((xs - land1) / STEP_LENGTH)
    h[in_second] = top + (k2[in_second] + 1) * step_height
    # chamfer: first sample row of each step, with probability 0.5
    step_id = np.where(in_first, k1, np.where(in_second, 1000 + k2, -1))
    first_row = np.zeros(n, bool)
    first_row[1:] = (step_id[1:] != step_id[:-1]) & (step_id[1:] >= 0)
    chamfer = {s: rng.uniform(-1, 1) < 0 for s in np.unique(step_id[first_row])}
    for i in np.nonzero(first_row)[0]:
        if chamfer[step_id[i]]:
            h[i] -= 0.025
    tile = np.repeat(h[:, None], n, axis=1)
    if noise_z_scale > 0:
        tile += fractal(n, hs, 20.0, noise_z_scale, rng)
    return tile


class TerrainBank:
    def __init__(self, tile_size=8.0, horizontal_scale=0.05, vertical_scale=0.005, num_variants=5,
                 border=3.0, stairs_noise=True, seed=0, device="cpu"):
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
                    sh = STEP_HEIGHT_LIST[level] * (1 if kind == KIND_UP else -1)
                    z = 0.2 if (kind == KIND_DOWN and stairs_noise) else 0.0
                    tile = stairs_tile(n, horizontal_scale, sh, rng, z)
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
        # robot spawn offset inside a tile (RaiSim: robot 0.6 m from the stairs map edge,
        # about 2 m from the rough map edge, laterally centred)
        self.spawn_local = torch.tensor([[2.0, self.tile_size / 2],
                                         [0.6, self.tile_size / 2],
                                         [0.6, self.tile_size / 2]], device=device)

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
