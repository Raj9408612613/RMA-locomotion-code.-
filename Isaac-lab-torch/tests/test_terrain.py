import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from terrain import TerrainBank, KIND_ROUGH, KIND_UP, KIND_DOWN, STEP_HEIGHT_LIST


def test_bank():
    tb = TerrainBank(num_variants=2, seed=1)
    # stairs up level 3: robot spawns on the 0.2 m platform, first step 1.2 m from tile start
    for kind, sign in [(KIND_UP, 1), (KIND_DOWN, -1)]:
        lvl = torch.tensor([1]); var = torch.tensor([1])
        xy = tb.spawn_xy(torch.tensor([kind]), lvl, var)
        h0 = tb.heights_at(xy).item()
        assert abs(h0 - 0.2) < (0.01 if kind == KIND_UP else 0.2), h0
        if kind == KIND_UP:
            # 0.9 m ahead of spawn (x = 1.5 local) is on step 1 at 0.2 + 0.13
            h1 = tb.heights_at(xy + torch.tensor([0.9, 0.0])).item()
            assert abs(h1 - (0.2 + STEP_HEIGHT_LIST[1])) < 0.01, h1
    # rough terrain amplitude is bounded
    xy = tb.spawn_xy(torch.tensor([KIND_ROUGH]), torch.tensor([0]), torch.tensor([0]))
    assert abs(tb.heights_at(xy).item()) < 0.3
    # bilinear lookup is exact on grid points and matches the mesh vertices
    verts, tris = tb.mesh()
    idx = np.random.default_rng(0).integers(0, len(verts), 1000)
    h = tb.heights_at(torch.tensor(verts[idx, :2])).numpy()
    assert np.allclose(h, verts[idx, 2], atol=1e-4)
    # all triangle normals point up
    v = verts[tris]
    nz = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 2]
    assert (nz > 0).all()
    print("mesh", verts.shape, tris.shape, "heights", tb.heights.shape)


if __name__ == "__main__":
    test_bank()
    print("ok")
