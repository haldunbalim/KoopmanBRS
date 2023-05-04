import numpy as np
from sklearn.decomposition import PCA
import polytope as pc
from utils.geometry import lims2hrep

def points_in(P, pts):
    return np.all(pts @ P.A.T <= P.b, axis=1)


def get_enclosing_oriented_box(pts, state_limits=None):
    if len(pts) < 2:
        return None
    if len(pts) == 2:
        return get_enclosing_aa_box(pts)
    pca = PCA(n_components=pts.shape[1])
    st_transformed = pca.fit_transform(pts)
    inf_max = np.max(st_transformed, axis=0)
    inf_min = np.min(st_transformed, axis=0)
    ct = (inf_max+inf_min)/2
    r2 = (inf_max-inf_min)/2
    domain = Zonotope(np.diag(r2), ct).affine_transform(
        pca.components_.T, pca.mean_)
    domain = domain.to_polytope()
    if state_limits is not None:
        domain = pc.reduce(domain.intersect(lims2hrep(state_limits)))
    return domain


def get_enclosing_aa_box(samples, state_limits=None, poly=True):
    lims = np.stack([samples.min(axis=0), samples.max(axis=0)], axis=0)
    domain = lims2hrep(lims)
    domain = domain
    if state_limits is not None:
        domain = pc.reduce(domain.intersect(lims2hrep(state_limits)))
    return domain


class ReachNode:
    def __init__(self, set_rep, A, B, error_bound_ct, error_bound_r, state_limits=None, action_limits=None,
                 error_set=None, parent_node=None) -> None:
        self.set_rep = set_rep
        self.A = A
        self.B = B
        self.error_bound_ct = error_bound_ct
        self.error_bound_r = error_bound_r
        self.error_set = error_set
        self.state_limits = state_limits
        self.action_limits = action_limits
        self.children = []
        self.parent = parent_node

    def split(self, ratio=0.5, *args, **kwargs):
        return self.set_rep.split(ratio=ratio, *args, **kwargs)

    def sample_pt(self, n_pts):
        return self.set_rep.sample_pt(n_pts)

    @property
    def G(self):
        return self.set_rep.G

    @property
    def c(self):
        return self.set_rep.c
    

class Zonotope:
    def __init__(self, generators, center=None):
        assert generators.ndim == 2, "Generators must be a matrix"
        self.G = generators
        if center is not None:
            assert center.ndim == 1, "Center must be a vector"
            assert center.shape[0] == generators.shape[0], "Center and generators must have the same dimension"
            self.c = center
        else:
            self.c = np.zeros(generators.shape[0])

    def affine_transform(self, A, b=None):
        center = A@self.c
        if b is not None:
            center += b
        return Zonotope(A@self.G, center)
    
    def to_polytope(self):
        V = self.c[:, np.newaxis]
        for g in self.G.T[..., np.newaxis]:
            V = np.concatenate([V+g, V-g], axis=1)
        return pc.reduce(pc.qhull(V.T))
