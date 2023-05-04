import numpy as np
import polytope as pc

def minkowski_diff_poly_box(P, r, ct=None):
    # r is half edge len of inf ball
    if type(P) == pc.Polytope:
        P = (P.A, P.b)
    P_A, P_b = P
    if type(r) == float:
        P_b_new = P_b - np.sum(np.abs(P_A), axis=1) * r
    else:
        P_b_new = P_b - np.abs(P_A) @ r

    if ct is not None:
        P_b_new -= P_A @ ct
    return pc.reduce(pc.Polytope(P_A, P_b_new))

def minkowski_sum_poly_box(P, r, ct=None):
    return minkowski_sum(P, get_inf_ball(r, P.A.shape[1], center=ct))

def lims2hrep(lims, center=None):
    p = pc.Polytope.from_box(lims.T)
    A, b = p.A[p.b != np.inf], p.b[p.b != np.inf]
    if center is not None:
        b = b + A @ center
    return pc.Polytope(A, b)

def get_inf_ball(r, nx, center=None):
    return lims2hrep(np.array([[-r]*nx, [r]*nx]), center=center)


def tulip_pre_proj(target_set, A, B, U, X=None):
    if type(U) != pc.Polytope:
        U = pc.Polytope(*U)
    nz = A.shape[1]
    X_A = np.concatenate([target_set.A@A, target_set.A@B], axis=1)
    U_A = np.concatenate([np.zeros((U.b.shape[0], nz)), U.A], axis=1)
    XU_A = np.concatenate([X_A, U_A], axis=0)
    XU_b = np.concatenate([target_set.b, U.b], axis=0)
    if X is not None:
        nu = B.shape[1]
        XU_A = np.concatenate([XU_A, np.concatenate([X.A,
                                                     np.zeros((X.A.shape[0], nu))],
                                                    axis=1)], axis=0)
        XU_b = np.concatenate([XU_b, X.b], axis=0)
    XU = pc.Polytope(XU_A, XU_b)
    return pc.projection(XU, list(range(1, nz+1)), solver="fm")

def minkowski_sum(P, S):
    if type(P) == pc.Polytope:
        P = (P.A, P.b)
    if type(S) == pc.Polytope:
        S = (S.A, S.b)
    P_A, P_b = P
    S_A, S_b = S
    Z = np.zeros((P_A.shape[0], S_A.shape[1]))
    A_u = np.concatenate([S_A, -S_A], axis=1)
    A_l = np.concatenate([Z, P_A], axis=1)
    A = np.concatenate([A_u, A_l], axis=0)
    b = np.concatenate([S_b, P_b], axis=0)
    return pc.projection(pc.Polytope(A, b), list(range(1, P_A.shape[1]+1)))
    

def lift_polytope(P, nd):
    return pc.Polytope(np.block([P.A, np.zeros((P.A.shape[0], nd))]), P.b)

def polytope_split(P, ratio=0.5, over_dims=None):
    if ratio != 0.5:
        raise NotImplementedError("Only ratio=0.5 is implemented")
    l, u = pc.bounding_box(P)
    l, u = l[:, 0], u[:, 0]
    c = (l+u)/2
    if over_dims is not None and over_dims != []:
        non_over_dims = np.setdiff1d(np.arange(P.A.shape[1]), over_dims)
        u[non_over_dims] = 0
        l[non_over_dims] = 0
        c[non_over_dims] = 0
    principal_axis = np.zeros(len(l))
    principal_axis[np.argmax(u-l)] = 1
    c1 = pc.Polytope(np.concatenate([P.A, principal_axis[np.newaxis, :]]),
                     np.concatenate([P.b, [principal_axis@c]]))
    c2 = pc.Polytope(np.concatenate([P.A, -principal_axis[np.newaxis, :]]),
                     np.concatenate([P.b, [-principal_axis@c]]))
    return c1, c2


    
