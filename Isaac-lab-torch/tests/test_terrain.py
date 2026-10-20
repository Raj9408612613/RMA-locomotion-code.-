import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from terrain import TerrainBank, KIND_ROUGH, KIND_UP, KIND_DOWN, STEP_HEIGHT_LIST


def test_bank():
    tb = TerrainBank(num_variants=2, seed=1)
    n_steps = 6  # Isaac Lab pyramid on an 8 m tile: (8 - 2 - 3) // 0.6 + 1
    for kind, sign in [(KIND_UP, -1), (KIND_DOWN, 1)]:
        lvl = 1
        h = STEP_HEIGHT_LIST[lvl]
        xy = tb.spawn_xy(torch.tensor([kind]), torch.tensor([lvl]), torch.tensor([1]))
        # spawn on the centre platform at +-(n_steps + 1) h
        assert abs(tb.heights_at(xy).item() - sign * (n_steps + 1) * h) < 1e-3, tb.heights_at(xy).item()
        # 1.35 m ahead: 2.65 m from the tile edge -> ring 5 at +-6h; 3.5 m ahead: flat border at 0
        assert abs(tb.heights_at(xy + torch.tensor([1.35, 0.0])).item() - sign * 6 * h) < 1e-3
        assert abs(tb.heights_at(xy + torch.tensor([3.5, 0.0])).item()) < 1e-3
    # RaiSim rough terrain: heights in [0, 0.22], fine bumps (std ~1.6 cm)
    xy = tb.spawn_xy(torch.tensor([KIND_ROUGH]), torch.tensor([0]), torch.tensor([0]))
    assert 0.0 <= tb.heights_at(xy).item() <= 0.22
    # bilinear lookup is exact on grid points and matches the mesh vertices
    verts, tris = tb.mesh()
    idx = np.random.default_rng(0).integers(0, len(verts), 1000)
    h = tb.heights_at(torch.tensor(verts[idx, :2])).numpy()
    assert np.allclose(h, verts[idx, 2], atol=1e-4)
    v = verts[tris]
    assert (np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 2] > 0).all()
    parts = tb.mesh_parts()
    assert parts[0][2] - 1 == parts[1][1] and parts[1][2] == tb.height_field.shape[0]  # one shared row
    print("mesh", verts.shape, tris.shape, "heights", tb.heights.shape)


if __name__ == "__main__":
    test_bank()
    print("ok")
