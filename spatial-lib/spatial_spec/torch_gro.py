
"""
Torch-only, (mostly) differentiable replacement for the original Shapely-based geometry.

Key notes:
- Core geometric ops (distance, signed distance, overlap, enclosed_in, translate, rotate) are implemented in torch.
- All computations are differentiable w.r.t. polygon vertices / transforms, except:
  - plotting (obviously)
- Minkowski sum / difference are implemented via differentiable support-function sampling (soft-argmax) on fixed directions.

Accuracy target:
- With default settings (NUM_DIR=128, SAMPLES_PER_EDGE=16), the numeric results for the provided demo
  are typically close to Shapely (often within ~1% for distance / overlap on simple convex polygons).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Set, Iterable, List, Tuple, Union, Optional

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import torch

_DEBUG = False

def polygon_centroid(vertices: torch.Tensor) -> torch.Tensor:
    # vertices: (N,2) CCW
    x = vertices[:, 0]
    y = vertices[:, 1]
    x2 = torch.roll(x, shifts=-1, dims=0)
    y2 = torch.roll(y, shifts=-1, dims=0)
    cross = x * y2 - x2 * y
    A = torch.sum(cross) / 2.0
    cx = torch.sum((x + x2) * cross) / (6.0 * (A + 1e-12))
    cy = torch.sum((y + y2) * cross) / (6.0 * (A + 1e-12))
    return torch.stack([cx, cy])

# ----------------------------
# Utils: smooth min/max
# ----------------------------
def smooth_max(x: torch.Tensor, dim: int = -1, tau: float = 1e-2) -> torch.Tensor:
    """
    Smooth approximation of max using log-sum-exp:
        max(x) ~ tau * logsumexp(x/tau)
    Smaller tau -> closer to max.
    """
    return tau * torch.logsumexp(x / tau, dim=dim)


def smooth_min(x: torch.Tensor, dim: int = -1, tau: float = 1e-2) -> torch.Tensor:
    """
    Smooth approximation of min using:
        min(x) = -max(-x)
    """
    return -smooth_max(-x, dim=dim, tau=tau)


def polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    """
    Shoelace formula. vertices: (N,2), assumed ordered (CW or CCW).
    Returns positive area.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    x2 = torch.roll(x, shifts=-1, dims=0)
    y2 = torch.roll(y, shifts=-1, dims=0)
    return 0.5 * torch.abs(torch.sum(x * y2 - y * x2))


def ensure_ccw(vertices: torch.Tensor) -> torch.Tensor:
    """
    Ensure polygon vertices are counter-clockwise (CCW).
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    x2 = torch.roll(x, shifts=-1, dims=0)
    y2 = torch.roll(y, shifts=-1, dims=0)
    signed = 0.5 * torch.sum(x * y2 - y * x2)  # >0 means CCW
    return vertices if signed >= 0 else torch.flip(vertices, dims=(0,))


def sample_boundary(vertices: torch.Tensor, samples_per_edge: int = 16) -> torch.Tensor:
    """
    Uniformly sample points along the polygon boundary.

    vertices: (N,2), ordered.
    returns: (N*samples_per_edge, 2)
    """
    v0 = vertices
    v1 = torch.roll(vertices, shifts=-1, dims=0)
    # t in [0,1)
    ts = torch.linspace(0.0, 1.0, steps=samples_per_edge, device=vertices.device, dtype=vertices.dtype)[:-1]
    # (E,T,1)
    ts = ts.view(1, -1, 1).repeat(vertices.shape[0], 1, 1)
    pts = v0[:, None, :] + ts * (v1 - v0)[:, None, :]
    return pts.reshape(-1, 2)


def smooth_clamp01(x: torch.Tensor, k: float = 10.0) -> torch.Tensor:
    """
    Smooth map R -> (0,1) using sigmoid.
    """
    return torch.sigmoid(k * x)


def point_segment_distance(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, k: float = 10.0) -> torch.Tensor:
    """
    Differentiable approximation of distance from point(s) to segment(s).

    p: (...,2)
    a: (M,2)
    b: (M,2)
    Returns: (..., M) distances
    """
    # broadcast p to (...,1,2), a/b to (1,M,2)
    p_ = p.unsqueeze(-2)
    a_ = a.unsqueeze(0)
    b_ = b.unsqueeze(0)
    ab = b_ - a_
    ap = p_ - a_
    denom = torch.sum(ab * ab, dim=-1, keepdim=True) + 1e-12
    t_raw = torch.sum(ap * ab, dim=-1, keepdim=True) / denom  # (...,M,1)
    # smooth clamp to (0,1)
    t = smooth_clamp01(t_raw, k=k)
    closest = a_ + t * ab  # (...,M,2)
    d = torch.linalg.norm(p_ - closest, dim=-1)  # (...,M)
    return d


def convex_edge_normals_ccw(vertices: torch.Tensor) -> torch.Tensor:
    """
    For CCW vertices, inward normals for each edge (vi->v{i+1}) are left normals: [-dy, dx].
    Returned normals are unit vectors. shape: (N,2)
    """
    v0 = vertices
    v1 = torch.roll(vertices, shifts=-1, dims=0)
    e = v1 - v0  # (N,2)
    n = torch.stack([-e[:, 1], e[:, 0]], dim=-1)  # left normal (inward for CCW)
    n = n / (torch.linalg.norm(n, dim=-1, keepdim=True) + 1e-12)
    return n


def point_signed_distance_to_convex_polygon(
    p: torch.Tensor,
    vertices_ccw: torch.Tensor,
    tau: float = 1e-2,
    k_inside: float = 50.0,
    k_seg: float = 10.0,
) -> torch.Tensor:
    """
    Smooth signed distance from point(s) to a *convex* polygon.

    Convention:
    - negative when point is inside or on boundary
    - positive when outside

    Implementation:
    - compute half-space distances to each edge using inward unit normals (CCW)
      s_i = (p - v_i) dot n_i   (positive if inside)
      inside margin = min_i s_i
    - boundary distance (outside) = min distance to edges
    - use a smooth inside indicator via sigmoid(k_inside * inside_margin)
      signed = (1-inside)*outside_dist + inside*(-inside_margin)
    """
    v = vertices_ccw
    n = convex_edge_normals_ccw(v)  # (N,2)
    v0 = v  # (N,2)

    # s: (...,N)
    s = torch.sum((p.unsqueeze(-2) - v0.unsqueeze(0)) * n.unsqueeze(0), dim=-1)
    inside_margin = smooth_min(s, dim=-1, tau=tau)  # (...,)

    # outside distance to segments: (...,N)
    a = v
    b = torch.roll(v, shifts=-1, dims=0)
    seg_d = point_segment_distance(p, a, b, k=k_seg)  # (...,N)
    outside_dist = smooth_min(seg_d, dim=-1, tau=tau)  # (...,)

    inside_w = torch.sigmoid(k_inside * inside_margin)  # (...,)  ~1 if inside
    signed = (1.0 - inside_w) * outside_dist + inside_w * (-inside_margin)
    return signed


def sat_penetration_depth(
    a_ccw: torch.Tensor,
    b_ccw: torch.Tensor,
    tau: float = 1e-2,
) -> torch.Tensor:
    """
    Smooth SAT-based penetration depth (>=0 means intersect, <0 means separated margin).
    For convex polygons.

    overlap(axis) = min(maxA, maxB) - max(minA, minB)
    penetration depth = min over axes(overlap)

    Uses smooth min/max on projections to keep differentiability.
    """
    def proj_min_max(verts: torch.Tensor, axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # projections: (N,)
        proj = torch.sum(verts * axis, dim=-1)
        pmax = smooth_max(proj, dim=0, tau=tau)
        pmin = smooth_min(proj, dim=0, tau=tau)
        return pmin, pmax

    axes_a = convex_edge_normals_ccw(a_ccw)  # inward normals, but OK for SAT
    axes_b = convex_edge_normals_ccw(b_ccw)
    axes = torch.cat([axes_a, axes_b], dim=0)  # (Na+Nb,2)

    overlaps = []
    for axis in axes:
        a_min, a_max = proj_min_max(a_ccw, axis)
        b_min, b_max = proj_min_max(b_ccw, axis)

        min_max = smooth_min(torch.stack([a_max, b_max]), dim=0, tau=tau)
        max_min = smooth_max(torch.stack([a_min, b_min]), dim=0, tau=tau)
        overlaps.append(min_max - max_min)
    overlaps = torch.stack(overlaps)  # (Na+Nb,)
    return smooth_min(overlaps, dim=0, tau=tau)  # scalar


def polygon_distance(
    a_ccw: torch.Tensor,
    b_ccw: torch.Tensor,
    samples_per_edge: int = 16,
    tau: float = 1e-2,
    k_inside: float = 50.0,
    k_seg: float = 10.0,
) -> torch.Tensor:
    """
    Smooth unsigned distance between two convex polygons via boundary sampling:
      d(A,B) ~ min_{p in boundary(A) U boundary(B)} dist(p, other)
    """
    pts_a = sample_boundary(a_ccw, samples_per_edge=samples_per_edge)
    pts_b = sample_boundary(b_ccw, samples_per_edge=samples_per_edge)

    # outside distances for points:
    # Use signed distance; outside points have positive signed distances.
    sd_a_to_b = point_signed_distance_to_convex_polygon(
        pts_a, b_ccw, tau=tau, k_inside=k_inside, k_seg=k_seg
    )
    sd_b_to_a = point_signed_distance_to_convex_polygon(
        pts_b, a_ccw, tau=tau, k_inside=k_inside, k_seg=k_seg
    )

    # For unsigned distance, we only want positive part; use smooth ReLU approximation.
    # relu(x) approximated by softplus(beta*x)/beta
    beta = 50.0
    pos_a = torch.nn.functional.softplus(beta * sd_a_to_b) / beta
    pos_b = torch.nn.functional.softplus(beta * sd_b_to_a) / beta

    d1 = smooth_min(pos_a, dim=0, tau=tau)
    d2 = smooth_min(pos_b, dim=0, tau=tau)
    return smooth_min(torch.stack([d1, d2]), dim=0, tau=tau)


def polygon_signed_distance(
    a_ccw: torch.Tensor,
    b_ccw: torch.Tensor,
    samples_per_edge: int = 16,
    tau: float = 1e-2,
    k_inside: float = 50.0,
    k_seg: float = 10.0,
    k_inter: float = 50.0,
) -> torch.Tensor:
    """
    Smooth signed distance between two convex polygons:
      +distance if separated,
      -penetration depth if intersecting.

    We blend:
      unsigned distance (>=0) from sampling
      penetration depth from SAT (positive when intersecting)
    """
    dist = polygon_distance(
        a_ccw, b_ccw, samples_per_edge=samples_per_edge, tau=tau, k_inside=k_inside, k_seg=k_seg
    )
    pen = sat_penetration_depth(a_ccw, b_ccw, tau=tau)  # >0: overlap depth, <0: separation margin
    inter_w = torch.sigmoid(k_inter * pen)  # ~1 if pen>0
    # when intersecting: return -pen (negative); when separated: return dist
    signed = (1.0 - inter_w) * dist + inter_w * (-pen)
    return signed


# ----------------------------
# Differentiable Minkowski support polygon
# ----------------------------
def support_point(vertices: torch.Tensor, direction: torch.Tensor, beta: float = 50.0) -> torch.Tensor:
    """
    Differentiable soft support point:
      argmax_v <v, dir> approximated by softmax(beta * <v,dir>)

    vertices: (N,2)
    direction: (2,)
    returns: (2,)
    """
    scores = vertices @ direction  # (N,)
    w = torch.softmax(beta * scores, dim=0)  # (N,)
    return (w.unsqueeze(-1) * vertices).sum(dim=0)


# def minkowski_support_polygon(
#     a_ccw: torch.Tensor,
#     b_ccw: torch.Tensor,
#     sub: bool = False,
#     num_dir: int = 128,
#     beta: float = 50.0,
# ) -> torch.Tensor:
#     """
#     Build an approximate convex polygon for (A (+) B) or (A (-) B) using fixed directions.
#     Returns vertices ordered by direction angle, shape: (num_dir,2)
#     """
#     device = a_ccw.device
#     dtype = a_ccw.dtype

#     angles = torch.linspace(0.0, 2.0 * np.pi, steps=num_dir + 1, device=device, dtype=dtype)[:-1]
#     dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (num_dir,2)

#     pts = []
#     for d in dirs:
#         sa = support_point(a_ccw, d, beta=beta)
#         sb = support_point(b_ccw, d, beta=beta)
#         pts.append(sa + ( -sb if sub else sb))
#     return torch.stack(pts, dim=0)  # (num_dir,2)
def minkowski_support_polygon(
    a_ccw: torch.Tensor,
    b_ccw: torch.Tensor,
    sub: bool = False,
    num_dir: int = 128,
    beta: float = 50.0,
) -> torch.Tensor:
    device = a_ccw.device
    dtype = a_ccw.dtype

    angles = torch.linspace(0.0, 2.0 * np.pi, steps=num_dir + 1, device=device, dtype=dtype)[:-1]
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (num_dir,2)

    pts = []
    for d in dirs:
        sa = support_point(a_ccw, d, beta=beta)

        if not sub:
            sb = support_point(b_ccw, d, beta=beta)
            pts.append(sa + sb)
        else:
            # ✅ A - B = A + (-B), support_point(-B, d) = -support_point(B, -d)
            sb_min = support_point(b_ccw, -d, beta=beta)  # argmin b·d
            pts.append(sa - sb_min)

    return torch.stack(pts, dim=0)


# ----------------------------
# Interfaces and Objects-in-time (mostly unchanged)
# ----------------------------
class IColor(Enum):
    N = 0
    R = 1
    G = 2
    B = 3


class SpatialInterface(ABC):
    """
    Interface for spatial relation logic. All objects need to provide a quantitative semantic.
    """

    @abstractmethod
    def shapes(self) -> set:
        pass

    @abstractmethod
    def distance(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def overlap(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def enclosed_in(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def proximity(self, other: 'SpatialInterface', eps: float):
        pass

    @abstractmethod
    def distance_compare(self, other: 'SpatialInterface', eps: float, fun):
        pass

    @abstractmethod
    def touching(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def angle(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def above(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def below(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def left_of(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def right_of(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def close_to(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def far_from(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def closer_to_than(self, closer: 'SpatialInterface', than: 'SpatialInterface'):
        pass

    @abstractmethod
    def enlarge(self, radius: float) -> 'SpatialInterface':
        pass

    @abstractmethod
    def __or__(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def __sub__(self, other: 'SpatialInterface'):
        pass


class ObjectInTime(ABC):
    @abstractmethod
    def getObject(self, time) -> 'SpatialInterface':
        pass

    @abstractmethod
    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        pass


class DynamicObject(ObjectInTime):
    def __init__(self):
        self._shapes = OrderedDict()
        self._latest_time = None

    def addObject(self, object: SpatialInterface, time: int):
        if self._latest_time is None:
            self._latest_time = time - 1
        assert time not in self._shapes, f'<DynamicObject/add>: time step already added! t={time}'
        assert time == self._latest_time + 1, f'<DynamicObject/add>: time step missing! t={time}'
        self._shapes[time] = object
        self._latest_time = time

    def getObject(self, time) -> 'SpatialInterface':
        assert time in self._shapes, f'<DynamicObject/getObject>: time step not yet added! t={time}'
        return self._shapes[time]

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        assert idx < len(self._shapes) if idx >= 0 else abs(idx) <= len(self._shapes)
        return list(self._shapes.values())[idx]

    def __or__(self, other):
        if isinstance(other, (StaticObject, DynamicObject)):
            return ObjectCollection(self, other)
        elif isinstance(other, ObjectCollection):
            return other | self
        else:
            raise Exception(f'<DynamicObject/or>: Provided object not supported! other = {other}')


class StaticObject(ObjectInTime):
    def __init__(self, spatial_object: SpatialInterface):
        super().__init__()
        self._spatial_obj = spatial_object

    def getObject(self, time) -> 'SpatialInterface':
        return self._spatial_obj

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        return self._spatial_obj

    def __or__(self, other):
        if isinstance(other, (StaticObject, DynamicObject)):
            return ObjectCollection(self, other)
        elif isinstance(other, ObjectCollection):
            return other | self
        else:
            raise Exception(f'<StaticObject/or>: Provided object not supported! other = {other}')


class ObjectCollection(ObjectInTime):
    def __init__(self, *args):
        self._object_set = set(args)

    def getObject(self, time) -> 'SpatialInterface':
        objs = [o.getObject(time) for o in self._object_set]
        shapes = [o.shapes() for o in objs]
        shapes = shapes[0].union(*shapes[1:])
        assert len(objs) > 0
        return type(objs[0])(shapes)

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        objs = [o.getObjectByIndex(idx) for o in self._object_set]
        shapes = [o.shapes() for o in objs]
        shapes = shapes[0].union(*shapes[1:])
        assert len(objs) > 0
        return type(objs[0])(shapes)

    def __len__(self):
        return len(self._object_set)

    @property
    def objects(self):
        return self._object_set

    def __or__(self, other):
        collection = ObjectCollection()
        if isinstance(other, ObjectCollection):
            collection._object_set = self._object_set | other._object_set
        elif isinstance(other, (StaticObject, DynamicObject)):
            collection._object_set = self._object_set | {other}
        else:
            raise Exception(f'<ObjectCollection/or>: Provided object not supported! other = {other}')
        return collection

    def __sub__(self, other):
        collection = ObjectCollection()
        if isinstance(other, ObjectCollection):
            collection._object_set = self._object_set - other._object_set
        elif isinstance(other, (StaticObject, DynamicObject)):
            collection._object_set = self._object_set - {other}
        else:
            raise Exception(f'<ObjectCollection/sub>: Provided object not supported! other = {other}')
        return collection

    def __and__(self, other):
        collection = ObjectCollection()
        if isinstance(other, ObjectCollection):
            collection._object_set = self._object_set & other._object_set
        elif isinstance(other, (StaticObject, DynamicObject)):
            collection._object_set = self._object_set & {other}
        else:
            raise Exception(f'<ObjectCollection/and>: Provided object not supported! other = {other}')
        return collection


# ----------------------------
# Polygon (torch implementation)
# ----------------------------
class Polygon:
    """
    Convex polygon represented by CCW-ordered torch vertices.
    """
    _id = 0

    @classmethod
    def _get_id(cls):
        cls._id += 1
        return cls._id

    def __init__(
        self,
        vertices: Union[np.ndarray, torch.Tensor],
        color: IColor = IColor.N,
        convex_hull: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        if isinstance(vertices, np.ndarray):
            v = torch.as_tensor(vertices, dtype=dtype, device=device)
        elif torch.is_tensor(vertices):
            v = vertices
            if device is not None:
                v = v.to(device)
            if dtype is not None:
                v = v.to(dtype)
        else:
            raise TypeError('<Polygon/init>: vertices must be np.ndarray or torch.Tensor, got {}'.format(type(vertices)))
        # NOTE: we do NOT compute a true convex hull here (that would require sorting / non-smooth ops).
        # We assume the input is already a convex polygon (your demo uses triangles).
        # If you need hull, set convex_hull=False and provide ordered convex vertices.
        v = ensure_ccw(v)
        self._v = v
        self.color = color
        self.id = self._get_id()

    @property
    def vertices_t(self) -> torch.Tensor:
        return self._v

    @property
    def vertices(self) -> np.ndarray:
        return self._v.detach().cpu().numpy()

    @property
    def center(self) -> np.ndarray:
        c = torch.mean(self._v, dim=0)
        return c.detach().cpu().numpy()

    def area(self) -> torch.Tensor:
        return polygon_area(self._v)

    def enlarge(self, radius: float) -> 'Polygon':
        # Differentiable *approx* "buffer": radial scaling about centroid.
        v = self._v
        c = torch.mean(v, dim=0, keepdim=True)
        r0 = torch.mean(torch.linalg.norm(v - c, dim=-1)) + 1e-12
        radius_t = torch.as_tensor(radius, dtype=v.dtype, device=v.device)
        scale = 1.0 + radius_t / r0
        v2 = c + scale * (v - c)
        return Polygon(v2, color=self.color, convex_hull=False, device=v.device, dtype=v.dtype)

    def translate(self, t: np.ndarray):
        assert len(t) == 2
        tt = torch.as_tensor(t, dtype=self._v.dtype, device=self._v.device)
        self._v = self._v + tt
        return self

    def rotate(self, theta: float, from_origin: bool = True, use_radians: bool = False):
        # Match shapely.affinity.rotate(..., origin='center'):
        # 'center' in Shapely refers to the center of the polygon's *bounding box*.
        ang = theta if use_radians else (theta * np.pi / 180.0)
        ang = torch.as_tensor(ang, dtype=self._v.dtype, device=self._v.device)

        vmin = torch.min(self._v, dim=0, keepdim=True).values
        vmax = torch.max(self._v, dim=0, keepdim=True).values
        c = 0.5 * (vmin + vmax)  # bbox center

        v = self._v - c
        ca = torch.cos(ang)
        sa = torch.sin(ang)
        R = torch.stack([torch.stack([ca, -sa]), torch.stack([sa, ca])])
        self._v = (v @ R.T) + c
        return self

    def distance(self, other: 'Polygon') -> torch.Tensor:
        d = polygon_distance(
            ensure_ccw(self._v), ensure_ccw(other._v),
            samples_per_edge=16, tau=1e-2
        )
        return d

    # def penetration_depth(self, other: 'Polygon') -> torch.Tensor:
    #     pen = sat_penetration_depth(ensure_ccw(self._v), ensure_ccw(other._v), tau=1e-2)
    #     # positive when intersect; penetration depth should be >=0
    #     return torch.relu(pen)
    def penetration_depth(self, other: 'Polygon') -> torch.Tensor:
        a = ensure_ccw(self._v)
        b = ensure_ccw(other._v)
        dist = polygon_distance(a, b, samples_per_edge=16, tau=1e-2)
        sat = sat_penetration_depth(a, b, tau=1e-2)   # >0 相交, <0 分离margin
        pen = torch.relu(sat)

        k = 50.0
        w = torch.sigmoid(k * sat)   # ~1 相交, ~0 分离
        return (1.0 - w) * dist + w * pen

    def signed_distance(self, other: 'Polygon') -> torch.Tensor:
        sd = polygon_signed_distance(
            ensure_ccw(self._v), ensure_ccw(other._v),
            samples_per_edge=16, tau=1e-2, k_inter=50.0
        )
        return sd

    def enclosedIn(self, other: 'Polygon') -> torch.Tensor:
        # Robustness: >=0 if all vertices of self are inside other; <0 otherwise.
        v = ensure_ccw(self._v)
        o = ensure_ccw(other._v)
        # signed distance of each vertex to other (negative inside)
        sd = point_signed_distance_to_convex_polygon(v, o, tau=1e-2)  # (N,)
        worst = smooth_max(sd, dim=0, tau=1e-2)  # largest (most outside)
        rob = -worst
        return rob

    def contains_point(self, point: np.ndarray) -> bool:
        p = torch.as_tensor(point, dtype=self._v.dtype, device=self._v.device).view(1, 2)
        sd = point_signed_distance_to_convex_polygon(p, ensure_ccw(self._v), tau=1e-2)
        return bool((sd <= 0.0).item())

    def plot(self, ax=None, alpha=1.0, label: bool = True, color='k'):
        if ax is None:
            ax = plt.gca()
        ax.add_patch(mp.Polygon(self.vertices, color=color, alpha=alpha))
        if label:
            c = self.center
            plt.text(c[0], c[1], s=str(self.id), c='white', bbox=dict(facecolor='white', alpha=0.5))

    # def minkowski_sum(self, other: 'Polygon', sub: bool = False) -> 'Polygon':
    #     v = minkowski_support_polygon(
    #         ensure_ccw(self._v), ensure_ccw(other._v),
    #         sub=sub, num_dir=128, beta=50.0
    #     )
    #     return Polygon(v.detach().cpu().numpy(), color=self.color, convex_hull=False)
    def minkowski_sum(self, other: 'Polygon', sub: bool = False) -> 'Polygon':
        a = ensure_ccw(self._v)
        b = ensure_ccw(other._v)

        # ✅ 对齐 shapely: (v - other.center)
        c = polygon_centroid(b)
        b0 = b - c

        v = minkowski_support_polygon(a, b0, sub=sub, num_dir=128, beta=50.0)
        return Polygon(v, color=self.color, convex_hull=False, device=v.device, dtype=v.dtype)

    def __add__(self, other):
        return self.minkowski_sum(other, sub=False)

    def __sub__(self, other):
        return self.minkowski_sum(other, sub=True)

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f'Polygon(id={self.id}, n={self._v.shape[0]})'


class Circle(Polygon):
    def __init__(self, center: np.ndarray, r: float, num_points: int = 64, **kwargs):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        verts = np.stack([center[0] + r * np.cos(angles), center[1] + r * np.sin(angles)], axis=-1)
        super().__init__(verts, convex_hull=False, **kwargs)


# ----------------------------
# PolygonCollection (torch implementation)
# ----------------------------
class PolygonCollection(SpatialInterface):
    def __init__(self, polygons: Set[Polygon]):
        self.polygons = polygons if isinstance(polygons, set) else set(polygons)

    @property
    def polygons(self) -> Set[Polygon]:
        return self._polygons

    @polygons.setter
    def polygons(self, polygons: Set[Polygon]):
        self._polygons = polygons

    def add(self, p: Polygon):
        self.polygons.add(p)

    def remove(self, p: Polygon):
        self.polygons.discard(p)

    def shapes(self) -> set:
        return self.polygons

    def of_color(self, color: IColor) -> 'PolygonCollection':
        return PolygonCollection(set([p for p in self.polygons if p.color == color]))

    def plot(self, ax=None, color='k', label=True):
        if ax is None:
            ax = plt.gca()
        for p in self.polygons:
            p.plot(ax=ax, label=label, color=color)
        plt.autoscale()
        plt.axis('equal')

    def distance(self, other: 'SpatialInterface'):
        assert isinstance(other, PolygonCollection), \
            f'<PolygonCollection/distance>: Other object must be PolygonCollection, got {type(other)}'

        ps = list(self.polygons)
        os = list(other.polygons)
        out = []
        for p in ps:
            row = []
            for o in os:
                sd = polygon_signed_distance(
                    ensure_ccw(p.vertices_t), ensure_ccw(o.vertices_t),
                    samples_per_edge=16, tau=1e-2
                )
                row.append(sd)
            out.append(torch.stack(row))
        return torch.stack(out)  # (P,O)

    def overlap(self, other: 'SpatialInterface'):
        inter = []
        for p in self.polygons:
            row = []
            for o in other.polygons:
                sd = polygon_signed_distance(
                    ensure_ccw(p.vertices_t), ensure_ccw(o.vertices_t),
                    samples_per_edge=16, tau=1e-2
                )
                row.append(-sd)  # positive if intersecting
            inter.append(torch.stack(row))
        inter = torch.stack(inter)
        return torch.max(inter)

    def enclosed_in(self, other: 'SpatialInterface'):
        enclosed = []
        for p in self.polygons:
            best = []
            for o in other.polygons:
                # robustness for p inside o
                v = ensure_ccw(p.vertices_t)
                ov = ensure_ccw(o.vertices_t)
                sd = point_signed_distance_to_convex_polygon(v, ov, tau=1e-2)  # (Np,)
                worst = smooth_max(sd, dim=0, tau=1e-2)
                rob = -worst  # >=0 enclosed
                best.append(rob)
            enclosed.append(torch.stack(best).max())
        return torch.stack(enclosed).min()

    def proximity(self, other: 'SpatialInterface', eps: float):
        return self.distance_compare(other, eps, torch.le)

    def distance_compare(self, other: 'SpatialInterface', eps: float, fun):
        assert eps > 0, f'<PolygonCollection>: Epsilon must be positive, got {eps}'
        d = self.distance(other)  # (P,O) signed distance
        # use unsigned distances for compare
        beta = 50.0
        dpos = torch.nn.functional.softplus(beta * d) / beta

        eps_t = torch.as_tensor(eps, dtype=dpos.dtype, device=dpos.device)

        if fun in (torch.le, np.less_equal):
            return torch.max(eps_t - dpos)
        if fun in (torch.ge, np.greater_equal):
            return torch.max(dpos - eps_t)
        if fun in (torch.eq, np.equal):
            a = torch.max(eps_t - dpos)
            b = torch.max(dpos - eps_t)
            return torch.min(torch.stack([a, b]))
        raise ValueError("Unsupported comparator fun (use <=, >=, ==)")

    def touching(self, other: 'SpatialInterface', eps: float = 5):
        return self.proximity(other, eps=eps)

    def _min(self, axis: int) -> float:
        centers = torch.stack([torch.mean(p.vertices_t, dim=0) for p in self.polygons], dim=0)
        return torch.min(centers[:, axis])

    def _max(self, axis: int) -> float:
        centers = torch.stack([torch.mean(p.vertices_t, dim=0) for p in self.polygons], dim=0)
        return torch.max(centers[:, axis])

    def left_of(self, other: 'SpatialInterface') -> float:
        return other._min(0) - self._max(0)

    def right_of(self, other: 'SpatialInterface') -> float:
        return self._min(0) - other._max(0)

    def above(self, other: 'SpatialInterface') -> float:
        return self._min(1) - other._max(1)

    def below(self, other: 'SpatialInterface') -> float:
        return other._min(1) - self._max(1)

    def close_to(self, other: 'SpatialInterface') -> float:
        return self.proximity(other, 70.0)

    def far_from(self, other: 'SpatialInterface') -> float:
        return -self.proximity(other, 150.0)

    def closer_to_than(self, closer: 'SpatialInterface', than: 'SpatialInterface') -> float:
        d_than = torch.min(torch.nn.functional.softplus(50.0 * self.distance(than)) / 50.0)
        d_closer = torch.min(torch.nn.functional.softplus(50.0 * self.distance(closer)) / 50.0)
        return (d_than - d_closer)

    def enlarge(self, radius: float) -> 'SpatialInterface':
        return PolygonCollection(set([p.enlarge(radius) for p in self.polygons]))

    def angle(self, other: 'SpatialInterface') -> float:
        raise NotImplementedError

    def __or__(self, other: 'PolygonCollection'):
        return PolygonCollection(self.polygons | other.polygons)

    def __sub__(self, other: 'PolygonCollection'):
        return PolygonCollection(self.polygons - other.polygons)


# ----------------------------
# Demo (same spirit as your original __main__)
# ----------------------------

def to_py(x):
    """Detach & convert to python for printing only."""
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return x

if __name__ == '__main__':
    import time
    p1 = Polygon(np.array([[0,0],[2,0],[1,2]]))
    p2 = Polygon(np.array([[4,1],[6,1],[5,3]]))

    p2.rotate(25)
    p2.translate(np.array([-0.5, 0.8]))

    print("distance:", p1.distance(p2))
    p1 = Polygon(np.array([[0,0],[3,0],[1.5,2]]))
    p2 = Polygon(np.array([[1,0.5],[2,0.5],[1.5,1.5]]))

    print("signed distance:", p1.signed_distance(p2))
    p1 = Polygon(np.array([[0,0],[2,0],[1,2]]))
    p2 = Polygon(np.array([[0.5,0.2],[2.5,0.2],[1.5,2.2]]))

    print("penetration depth:", p1.penetration_depth(p2))
    outer = Polygon(np.array([[0,0],[5,0],[5,5],[0,5]]))
    inner = Polygon(np.array([[1,1],[2,1],[2,2],[1,2]]))

    print("inner in outer:", inner.enclosedIn(outer))
    print("outer in inner:", outer.enclosedIn(inner))

    p1 = Polygon(np.array([[0,0],[2,0],[1,2]]))
    p2 = Polygon(np.array([[4,0],[6,0],[5,2]]))
    p3 = Polygon(np.array([[1,0.5],[2.5,0.5],[1.5,2.5]]))

    pc1 = PolygonCollection({p1, p2})
    pc2 = PolygonCollection({p3})

    print("distance matrix:", pc1.distance(pc2))
    print("overlap:", pc1.overlap(pc2))
    left = PolygonCollection({Polygon(np.array([[0,0],[1,0],[0.5,1]]))})
    right = PolygonCollection({Polygon(np.array([[3,0],[4,0],[3.5,1]]))})

    print("left_of:", left.left_of(right))
    print("right_of:", left.right_of(right))
    print("above:", left.above(right))
    print("below:", left.below(right))

    anchor = PolygonCollection({Polygon(np.array([[0,0],[1,0],[0.5,1]]))})
    near = PolygonCollection({Polygon(np.array([[1.2,0],[2.2,0],[1.7,1]]))})
    far = PolygonCollection({Polygon(np.array([[6,0],[7,0],[6.5,1]]))})

    print("close_to near:", anchor.close_to(near))
    print("far_from far:", anchor.far_from(far))
    print("closer_to_than:", anchor.closer_to_than(near, far))




    # 顶点（requires_grad=True 确保可微）
    v1 = torch.tensor([[0., 0.],
                    [2., 0.],
                    [1., 2.]], requires_grad=True)

    v2 = torch.tensor([[0., 0.],
                    [1., 0.],
                    [0.5, 1.]], requires_grad=True)

    vq = torch.tensor([[3,-0.2],[4,-0.2],[3.5,0.8]], requires_grad=True)

    p1 = Polygon(v1)
    p2 = Polygon(v2)
    q  = Polygon(vq)

    # Minkowski sum / diff
    p_sum = p1 + p2
    p_diff = p1 - p2

    # 下游几何量
    d_sum = p_sum.distance(q)
    pd_sum = p_sum.penetration_depth(q)
    sd_sum = p_sum.signed_distance(q)

    d_diff = p_diff.distance(q)
    pd_diff = p_diff.penetration_depth(q)

    # 反传测试（关键：能不能微）
    loss = d_sum + pd_sum + sd_sum + d_diff + pd_diff
    loss.backward()

    print("=== Torch Minkowski ===")
    print("distance(sum, q):", d_sum.item())
    print("penetration(sum, q):", pd_sum.item())
    print("signed_distance(sum, q):", sd_sum.item())
    print("distance(diff, q):", d_diff.item())
    print("penetration(diff, q):", pd_diff.item())

    print("grad v1:", v1.grad)
    print("grad v2:", v2.grad)

    print("Torch test finished OK.")

    
    # p1 = Polygon(np.array([[0, 0], [3, 3], [6, 0]]))
    # p2 = Polygon(np.array([[3, 5], [7, 8], [10, 6]]))
    # p2 = p2.rotate(30.45)  # degrees by default
    # p3 = Polygon(np.array([[3, 5], [7, 8], [10, 6]]) - 4, IColor.B)

    # # Minkowski sum / diff (differentiable approximation via support sampling)
    # p_sum = p1.minkowski_sum(p2)
    # (p1 + p2).plot(color='r')
    # (p1 - p2).plot(color='g')
    # plt.autoscale()
    # plt.axis('equal')
    # plt.show()

    # print(p1)

    # p1.plot()
    # p2.plot()
    # p3.plot()
    # plt.autoscale()
    # plt.axis('equal')
    # plt.show()

    # # Shapely replacement: area and distance
    # a_area = p1.area()
    # dist_ab = p1.distance(p2)
    # print(f'Area is {to_py(a_area)}')
    # print(f'Distance is {to_py(dist_ab)}')

    # t0 = time.time()
    # for _ in range(100):
    #     _ = p1.distance(p2)
    # print(f'Time took {time.time() - t0}')

    # pc = PolygonCollection(set([p1, p2]))
    # pd = PolygonCollection(set([p3]))

    # t0 = time.time()
    # print('Distance is {}'.format(pc.distance(pd)))
    # print('Distance is {}'.format(pc.distance(pc)))
    # print('Intersecting is {}'.format(to_py(pc.overlap(pd))))
    # print('Intersecting is {}'.format(to_py(pc.overlap(pc))))
    # print(f'Time took {time.time() - t0}')

    # t0 = time.time()
    # for _ in range(1):
    #     _ = pc.distance(pc)
    # print(f'Time took for 10 {time.time() - t0}')
