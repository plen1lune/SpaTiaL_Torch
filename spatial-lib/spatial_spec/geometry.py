from abc import *
from collections import OrderedDict
from enum import Enum
from typing import Set

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity as af
import shapely.geometry as sh

_DEBUG = False


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
        """
        Returns the shapes stored in the SpatialInterface object
        Returns: The shapes of the SpatialInterface object

        """
        pass

    @abstractmethod
    def distance(self, other: 'SpatialInterface') -> float:
        """
        Returns the signed distance to another spatial interface object
        Args:
            other: The other spatial interface object

        Returns: Distance (squared) to other object

        """
        pass

    @abstractmethod
    def overlap(self, other: 'SpatialInterface') -> float:
        """
        Computes if this object overlaps with another object
        Args:
            other: The other object

        Returns: >=0 if both objects overlap and <0 otherwise

        """
        pass

    @abstractmethod
    def enclosed_in(self, other: 'SpatialInterface') -> float:
        """
        Computes if this objects is enclosed in another object. If any this object is a collection, every object
        must be enclosed in an object of other
        Args:
            other: The other object

        Returns: >=0 if this object is enclosed in the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def proximity(self, other: 'SpatialInterface', eps: float) -> bool:
        """
        Computes if this objects is in proximity to another object
        Args:
            other: The other object
            eps: Specification of proximity

        Returns: >=0 if objects are in proximity and <0 otherwise

        """
        pass

    @abstractmethod
    def distance_compare(self, other: 'SpatialInterface', eps: float, fun) -> bool:
        """
        Compares the distance between two objects and a target value (e.g., a dist b <= eps)
        Args:
            other: The other object
            eps: The target value
            fun: The function for comparing (<=,>=,==)

        Returns: >=0 if predicate is true and <0 otherwise

        """
        pass

    @abstractmethod
    def touching(self, other: 'SpatialInterface') -> bool:
        """
        Computes if two objects are touching
        Args:
            other: The other object

        Returns:

        """
        pass

    @abstractmethod
    def angle(self, other: 'SpatialInterface') -> bool:
        """
        Computes the angle between to objects
        Args:
            other: The other object

        Returns: NOT YET IMPLEMENTED / USED

        """
        pass

    @abstractmethod
    def above(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is above another object
        Args:
            other: The other object

        Returns: >= 0 if this object is above the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def below(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is below another object
        Args:
            other: The other object

        Returns: >= 0 if this object is below the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def left_of(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is left of another object
        Args:
            other: The other object

        Returns: >= 0 if this object is left of the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def right_of(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is right of another object
        Args:
            other: The other object

        Returns: >= 0 if this object is right of the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def close_to(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is close to another object
        Args:
            other: The other object

        Returns: >= 0 if this object is close to the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def far_from(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is far from another object
        Args:
            other: The other object

        Returns: >= 0 if this object is far from the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def closer_to_than(self, closer: 'SpatialInterface', than: 'SpatialInterface') -> bool:
        """
        Computes if this object is closer to one object than another
        Args:
            closer: The object that should be closer
            than: The object that should be further away

        Returns: >= 0 if this object is closer to one object than another and <0 otherwise

        """
        pass

    @abstractmethod
    def enlarge(self, radius: float) -> 'SpatialInterface':
        """
        Enlarges an object with a given radius
        Args:
            radius: The radius for enlarging the object

        Returns: The enlarged object

        """
        pass

    @abstractmethod
    def __or__(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def __sub__(self, other: 'SpatialInterface'):
        pass


class ObjectInTime(ABC):
    """
    Interface for an object changing with time.
    """

    @abstractmethod
    def getObject(self, time) -> 'SpatialInterface':
        """
        Returns the object at the given time point
        """
        pass

    @abstractmethod
    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        """
        Returns the object at the given time point (given as index)
        Args:
            idx: The index of the time step

        Returns:

        """
        pass


class DynamicObject(ObjectInTime):

    def __init__(self):
        self._shapes = OrderedDict()  # compatible with all Python versions, preserves insertion order
        self._latest_time = None

    def addObject(self, object: SpatialInterface, time: int):
        if self._latest_time is None:
            self._latest_time = time - 1
        assert time not in self._shapes, '<DynamicObject/add>: time step already added! t={}'.format(time)
        assert time == self._latest_time + 1, '<DynamicObject/add>: time step missing! t = {}'.format(time)
        self._shapes[time] = object
        self._latest_time = time

    def getObject(self, time) -> 'SpatialInterface':
        assert time in self._shapes, '<DynamicObject/add>: time step not yet added! t={}'.format(time)
        return self._shapes[time]

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        assert idx < len(self._shapes) if idx >= 0 else abs(idx) <= len(self._shapes)
        return list(self._shapes.values())[idx]

    def __or__(self, other):
        if isinstance(other, (StaticObject, DynamicObject)):
            return ObjectCollection(self, other)
        elif isinstance(other, ObjectCollection):
            return other + self
        else:
            raise Exception('<DynamicObject/add>: Provided object not supported! other = {}'.format(other))


class StaticObject(ObjectInTime):
    """
    An SpatialInterface object static in time. The simplest implementation of ObjectInTime
    """

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
            return other + self
        else:
            raise Exception('<DynamicObject/add>: Provided object not supported! other = {}'.format(other))


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
        shapes = [o.shapes for o in objs]
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
            raise Exception('<ObjectCollection/add>: Provided object not supported! other = {}'.format(other))
        return collection

    def __sub__(self, other):
        collection = ObjectCollection()
        if isinstance(other, ObjectCollection):
            collection._object_set = self._object_set - other._object_set
        elif isinstance(other, (StaticObject, DynamicObject)):
            collection._object_set = self._object_set - {other}
        else:
            raise Exception('<ObjectCollection/add>: Provided object not supported! other = {}'.format(other))
        return collection

    def __and__(self, other):
        collection = ObjectCollection()
        if isinstance(other, ObjectCollection):
            collection._object_set = self._object_set & other._object_set
        elif isinstance(other, (StaticObject, DynamicObject)):
            collection._object_set = self._object_set & {other}
        else:
            raise Exception('<ObjectCollection/add>: Provided object not supported! other = {}'.format(other))
        return collection


class Polygon(object):
    """
    Class representing a polygon
    """
    _id = 0
    _ORIGIN = sh.Point([0, 0])  # origin for penetration depth computation
    # _MinkowskiDiff = lambda a, b: sh.Polygon(np.vstack(np.repeat([a],len(b),axis=0)-b)).convex_hull
    _MinkowskiDiff = lambda a, b: sh.Polygon(np.vstack([a - v for v in b])).convex_hull

    @classmethod
    def _get_id(cls):
        cls._id += 1
        return cls._id

    def __init__(self, vertices: np.ndarray, color: IColor = IColor.N, convex_hull: bool = True):
        """
        Initializes a polygon object
        Args:
            vertices: The vertices of the polygon
            color: The color of the polygon. Default = IColor.N
            convex_hull: bool to set if convex hull should be computed. Default = True
        """

        assert isinstance(vertices, np.ndarray), '<Polygon/init>: vertices must be of type np.ndarray!'

        if convex_hull:
            self.shape = sh.Polygon(vertices).convex_hull
        else:
            self.shape = sh.Polygon(vertices)
        self.color = color
        self.id = self._get_id()

    @property
    def shape(self) -> sh.Polygon:
        """
        Returns the shapely polygon object of this polygon
        Returns: Shapely polygon

        """
        return self._shape

    @shape.setter
    def shape(self, shape: sh.Polygon):
        """
        Sets the shapely polygon of this polygon
        Args:
            shape: The new shapely polygon

        """
        assert isinstance(shape, sh.Polygon), '<Polygon/shape>: Only shapely polygons are supported'
        self._shape = shape

    @property
    def vertices(self) -> np.ndarray:
        """
        Returns the vertices of the polygon
        Returns: The vertices of the polygon as a numpy array

        """
        return np.array(self.shape.exterior.coords)

    @property
    def center(self) -> np.ndarray:
        """
        Returns the geometric center of the polygon
        Returns: Geometric center of the polygon as a numpy array

        """
        return np.array(self.shape.centroid.coords[0])

    def enlarge(self, radius: float) -> 'Polygon':
        enlarged = self.shape.buffer(radius)
        return Polygon(np.array(enlarged.exterior.coords))

    def translate(self, t: np.ndarray):
        """
        Translates the polygon by the given translation vector
        Args:
            t: Translation vector as numpy array with shape (2x1)

        Returns: Translated version of this polygon (no copy)

        """

        assert len(t) == 2
        self.shape = af.translate(self.shape, t[0], t[1])
        return self

    def rotate(self, theta: float, from_origin: bool = True, use_radians=False):
        """
        Rotates the polygon around its center (of its bounding box)
        Args:
            theta: The angle of the rotation
            from_origin: currently not used
            use_radians: True if angle is given in radian

        Returns: Rotated version of this polygon (no copy)

        """

        self.shape = af.rotate(self.shape, theta, origin='center', use_radians=use_radians)
        return self

    def distance(self, other: 'Polygon'):
        """
        Computes the distance to another polygon object
        Args:
            other: The other polygon object

        Returns: The distance (>=0) between this and the other object

        """
        return self.shape.distance(other.shape)

    def penetration_depth(self, other: 'Polygon'):
        """
        Computes the penetration depth with another polygon object
        Args:
            other: The other polygon object

        Returns: The penetration depth (>=0) between this and the other object.
        Zero if no intersection between the objects.

        """
        # return Polygon._MinkowskiDiff(np.asarray(self.shape.exterior.coords),
        #                              np.asarray(other.shape.exterior.coords)).exterior.distance(self._ORIGIN)
        return self._penetration_depth(np.asarray(self.shape.exterior.coords), np.asarray(other.shape.exterior.coords))

    def _penetration_depth(self, vert1: np.ndarray, vert2: np.ndarray):
        return Polygon._MinkowskiDiff(vert1, vert2).exterior.distance(self._ORIGIN)

    def signed_distance(self, other: 'Polygon'):
        """
        Computes the signed distance of this polygon to another one
        Args:
            other: The other polygon

        Returns: The signed distance between the two polygons (<= 0 if touching/intersection, >0 if no penetration)

        """
        gjk = self.distance(other)
        return gjk - self.penetration_depth(other) if gjk <= 0.0000001 else gjk

    def enclosedIn(self, other: 'Polygon'):
        """
        Computes if this polygon is enclosed in another polygon (i.e., all vertices are have negative signed distance)
        Args:
            other: The superset polygon

        Returns: >=0 if this polygon is enclosed in the other polygon, <0 otherwise

        """
        sd = -np.inf
        o = np.array(other.shape.exterior.coords)
        for v in self.vertices:
            gjk = other.shape.distance(sh.Point(v))
            sd_c = gjk - self._penetration_depth(v, o) if gjk < 0.0000001 else gjk
            if sd_c > sd:
                sd = sd_c
        return -sd if not np.isclose(sd, 0) else sd

    def contains_point(self, point: np.ndarray):
        """
        Checks whether a given point is enclosed in the polygon
        Args:
            point: The point to check

        Returns: True if the point is enclosed in the polygon and False otherwise

        """
        return self.shape.contains(sh.Point(point))

    @property
    def color(self) -> IColor:
        """
        Color of polygon
        Returns: Color of polygon

        """
        return self._color

    @color.setter
    def color(self, color: IColor):
        """
        Color of polygon
        Args:
            color: New color of circle

        """
        self._color = color

    def plot(self, ax=None, alpha=1.0, label: bool = True, color='k'):
        """
        Plots the polygon
        Args:
            ax: The axis object to plot to (if provided)
            alpha: The alpha value of the circle
            label: bool to indicate whether to plot label

        """

        if ax is None:
            ax = plt.gca()

        ax.add_patch(
            mp.Polygon(self.vertices, color=color, alpha=alpha))
        if label:
            plt.text(self.center[0], self.center[1], s=str(self.id), c='white', bbox=dict(facecolor='white', alpha=0.5))

    def minkowski_sum(self, other: 'Polygon', sub: bool = False) -> 'Polygon':
        new_vertices = list()
        for v in other.vertices:
            if not sub:
                new_vertices.append(self.vertices + (v - other.center))
            else:
                new_vertices.append(self.vertices - (v - other.center))
        return Polygon(np.vstack(new_vertices))

    def __add__(self, other):
        return self.minkowski_sum(other)

    def __sub__(self, other):
        return self.minkowski_sum(other, sub=True)

    def __hash__(self):
        return self.id


class Circle(Polygon):

    def __init__(self, center: np.ndarray, r: float):
        # approximate circle
        vertices = np.array(sh.Point(center).buffer(r).exterior.coords)
        super().__init__(vertices, convex_hull=False)


class PolygonCollection(SpatialInterface):
    """
        Implements spatial interface for objects of type polytope. Represents set of polytopes
        """

    def __init__(self, polygons: Set[Polygon]):
        """
        Initializes a circle collection with a set of circles
        Args:
            circles: Set of circles
        """
        self.polygons = polygons if isinstance(polygons, set) else set(polygons)

    @property
    def polygons(self) -> Set[Polygon]:
        """
        Set of polygons
        Returns: set of polytopes

        """
        return self._polygons

    @polygons.setter
    def polygons(self, polygons: Set[Polygon]):
        """
        Set of polygons
        Args:
            polygons: new set of polytopes

        Returns:

        """
        self._polygons = polygons

    def add(self, p: Polygon):
        """
        Adds a polygons object to this collection
        Args:
            p: The polygons to add


        """
        self.polygons.add(p)

    def remove(self, p: Polygon):
        """
        Removes a polygons from this collection
        Args:
            p: The polygons to remove


        """
        self.polygons.discard(p)

    def shapes(self) -> set:
        return self.polygons

    def of_color(self, color: IColor) -> 'PolygoneCollection':
        """
        Returns a polygons collection containing polytopes of the specified color
        Args:
            color: The specified color

        Returns: polygons collection containing polytopes of specific color

        """
        return PolygonCollection(set([p for p in self.polygons if p.color == color]))

    def plot(self, ax=None, color='k', label=True):
        """
        Draws all polygons in this collection
        Args:
            ax: The axis object to plot to
            label: bool to indicate whether to plot labels

        Returns:

        """
        if ax is None:
            ax = plt.gca()
        for p in self.polygons:
            p.plot(ax=ax, label=label, color=color)
        plt.autoscale()
        plt.axis('equal')

    def distance(self, other: 'SpatialInterface') -> float:
        assert isinstance(other, PolygonCollection), \
            '<Polygon/distance>: Other object must be of type polygon, got {}'.format(other)

        # compute distances

        result = list()
        for p in self.polygons:
            result.append([p.signed_distance(o) for o in other.polygons])

        return result

    def overlap(self, other: 'SpatialInterface') -> bool:
        # intersection polygons
        inter = list()
        for p in self.polygons:
            inter.append([-p.signed_distance(o) for o in other.polygons])
        inter = np.array(inter)

        return np.max(inter)

    def enclosed_in(self, other: 'SpatialInterface') -> float:
        enclosed = list()
        for p in self.polygons:
            enclosed.append(np.array([p.enclosedIn(o) for o in other.polygons]).max())
        return np.array(enclosed).min()

    def proximity(self, other: 'SpatialInterface', eps: float) -> bool:
        return self.distance_compare(other, eps, np.less_equal)

    def distance_compare(self, other: 'SpatialInterface', eps: float, fun):
        assert np.positive(eps), '<Polygon>: Epsilon must be positive, got {}'.format(eps)

        # compute result
        if fun == np.less_equal:
            return np.max(np.repeat(eps, len(other.polygons)) - self.distance(other))
        if fun == np.greater_equal:
            return np.max(self.distance(other) - np.repeat(eps, len(other.polygons)))
        if fun == np.equal:
            return np.min([np.max(np.repeat(eps, len(other.polygons)) - self.distance(other)),
                           np.max(self.distance(other) - np.repeat(eps, len(other.polygons)))])

    def touching(self, other: 'SpatialInterface', eps: float = 5) -> bool:
        return self.proximity(other, eps=eps)
        return np.min([self.proximity(other, eps=eps), -self.proximity(other, eps=-eps)])

    def _min(self, axis: int) -> float:
        """
        Returns the minimum value of the projection of all polygons to the specified axis
        Args:
            axis: The specified axis

        Returns: The minimum value along the specified axis

        """
        return np.min([c.center[axis] for c in self.polygons])

    def _max(self, axis: int) -> float:
        """
        Returns the maximum value of the projection of all polygons to the specified axis
        Args:
            axis: The specified axis

        Returns: The maximum value along the specified axis

        """
        return np.max([c.center[axis] for c in self.polygons])

    def left_of(self, other: 'SpatialInterface') -> float:
        return other._min(0) - self._max(0)

    def right_of(self, other: 'SpatialInterface') -> float:
        return self._min(0) - other._max(0)

    def above(self, other: 'SpatialInterface') -> float:
        return self._min(1) - other._max(1)

    def below(self, other: 'SpatialInterface') -> float:
        return other._min(1) - self._max(1)

    def close_to(self, other: 'SpatialInterface') -> float:
        return self.proximity(other, 70.)

    def far_from(self, other: 'SpatialInterface') -> float:
        return -self.proximity(other, 150)

    def closer_to_than(self, closer: 'SpatialInterface', than: 'SpatialInterface') -> float:
        return np.min(self.distance(than)) - np.min(self.distance(closer))

    def enlarge(self, radius: float) -> 'SpatialInterface':
        return PolygonCollection(set([p.enlarge(radius) for p in self.polygons]))

    def angle(self, other: 'CircleOLD') -> float:
        pass

    def __or__(self, other: 'PolytopeCollection'):
        return PolygonCollection(self.polygons | other.polygons)

    def __sub__(self, other: 'PolytopeCollection'):
        return PolygonCollection(self.polygons - other.polygons)


if __name__ == '__main__':
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
    
    # p1 = Polygon(np.array([[0, 0], [3, 3], [6, 0]]))
    # p2 = Polygon(np.array([[3, 5], [7, 8], [10, 6]]))
    # p2 = p2.rotate(30.45)
    # p3 = Polygon(np.array([[3, 5], [7, 8], [10, 6]]) - 4, IColor.B)

    # p_sum = p1.minkowski_sum(p2)
    # (p1 + p2).plot(color='r')
    # (p1 - p2).plot(color='g')
    # plt.autoscale()
    # plt.show()
    # print(p1)

    # p1.plot()
    # p2.plot()
    # p3.plot()
    # plt.autoscale()
    # plt.show()
    # import time

    # a = sh.Polygon(p1.vertices)
    # b = sh.Polygon(p2.vertices)
    # print('Area is {}'.format(a.area))
    # print('Distance is {}'.format(a.distance(b)))

    # t0 = time.time()
    # for i in range(100):
    #     a.distance(b)
    # print(f'Time took {time.time() - t0}')

    # pc = PolygonCollection(set([p1, p2]))
    # pd = PolygonCollection(set([p3]))

    # t0 = time.time()
    # print('Distance is {}'.format(pc.distance(pd)))
    # print('Distance is {}'.format(pc.distance(pc)))
    # print('Intersecting is {}'.format(pc.overlap(pd)))
    # print('Intersecting is {}'.format(pc.overlap(pc)))
    # print(f'Time took {time.time() - t0}')

    # t0 = time.time()
    # for i in range(1):
    #     pc.distance(pc)
    #     # p1.intersect(p2).volume
    # print(f'Time took for 10 {time.time() - t0}')
    p1 = Polygon(np.array([[0, 0], [2, 0], [1, 2]]))
    p2 = Polygon(np.array([[0, 0], [1, 0], [0.5, 1]]))
    q  = Polygon(np.array([[3,-0.2],[4,-0.2],[3.5,0.8]]))

    # Minkowski sum / diff
    p_sum = p1 + p2
    p_diff = p1 - p2

    # 下游几何量（你代码里真实用到的）
    d_sum = p_sum.distance(q)
    pd_sum = p_sum.penetration_depth(q)
    sd_sum = p_sum.signed_distance(q)

    d_diff = p_diff.distance(q)
    pd_diff = p_diff.penetration_depth(q)

    print("=== Shapely Minkowski ===")
    print("distance(sum, q):", d_sum)
    print("penetration(sum, q):", pd_sum)
    print("signed_distance(sum, q):", sd_sum)
    print("distance(diff, q):", d_diff)
    print("penetration(diff, q):", pd_diff)

    print("Shapely test finished OK.")

# from abc import *
# from collections import OrderedDict
# from enum import Enum
# from typing import Set, Iterable, List, Tuple, Optional

# import matplotlib.patches as mp
# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# _DEBUG = False


# # =========================
# # Torch geometry utilities
# # =========================

# def _to_torch_xy(xy) -> torch.Tensor:
#     """
#     Convert input to torch tensor with shape (N,2) float64 on CPU.
#     """
#     if isinstance(xy, torch.Tensor):
#         t = xy
#     else:
#         t = torch.as_tensor(np.asarray(xy), dtype=torch.float64)
#     if t.ndim == 1:
#         t = t.view(1, 2)
#     assert t.shape[-1] == 2, f"Expected (...,2), got {t.shape}"
#     return t.to(dtype=torch.float64, device="cpu")


# def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     # a,b: (...,2) -> (...,)
#     return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# def _dot2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     return (a * b).sum(dim=-1)


# def _segment_distance_point(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """
#     Distance from point p to segment ab.
#     p: (2,), a:(2,), b:(2,)
#     """
#     ab = b - a
#     ap = p - a
#     denom = _dot2(ab, ab).clamp_min(1e-18)
#     t = (_dot2(ap, ab) / denom).clamp(0.0, 1.0)
#     proj = a + t * ab
#     return torch.linalg.norm(p - proj)


# def _segments_intersect(a1: torch.Tensor, a2: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor) -> bool:
#     """
#     Proper segment intersection test (including collinear overlap).
#     """
#     # Using orientation tests
#     def orient(p, q, r):
#         return _cross2(q - p, r - p)

#     def on_segment(p, q, r):
#         # q on pr bounding box (collinear)
#         return (torch.min(p[0], r[0]) - 1e-12 <= q[0] <= torch.max(p[0], r[0]) + 1e-12 and
#                 torch.min(p[1], r[1]) - 1e-12 <= q[1] <= torch.max(p[1], r[1]) + 1e-12)

#     o1 = orient(a1, a2, b1)
#     o2 = orient(a1, a2, b2)
#     o3 = orient(b1, b2, a1)
#     o4 = orient(b1, b2, a2)

#     # General case
#     if (o1 * o2 < 0) and (o3 * o4 < 0):
#         return True

#     # Collinear cases
#     if torch.isclose(o1, torch.tensor(0.0, dtype=torch.float64)) and on_segment(a1, b1, a2):
#         return True
#     if torch.isclose(o2, torch.tensor(0.0, dtype=torch.float64)) and on_segment(a1, b2, a2):
#         return True
#     if torch.isclose(o3, torch.tensor(0.0, dtype=torch.float64)) and on_segment(b1, a1, b2):
#         return True
#     if torch.isclose(o4, torch.tensor(0.0, dtype=torch.float64)) and on_segment(b1, a2, b2):
#         return True

#     return False


# def _polygon_edges(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     v: (N,2) -> (N,2) start and end arrays for edges.
#     """
#     n = v.shape[0]
#     a = v
#     b = torch.roll(v, shifts=-1, dims=0)
#     return a, b


# def _point_in_polygon(point: torch.Tensor, poly: torch.Tensor) -> bool:
#     """
#     Ray casting algorithm for simple polygon. Returns True if inside (strict-ish).
#     point: (2,), poly: (N,2)
#     """
#     x, y = float(point[0]), float(point[1])
#     verts = poly
#     n = verts.shape[0]
#     inside = False
#     for i in range(n):
#         x1, y1 = float(verts[i, 0]), float(verts[i, 1])
#         x2, y2 = float(verts[(i + 1) % n, 0]), float(verts[(i + 1) % n, 1])
#         # Check if edge crosses ray
#         if ((y1 > y) != (y2 > y)):
#             xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1
#             if x < xinters:
#                 inside = not inside
#     return inside


# def _polygon_area_and_centroid(poly: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Returns (area, centroid) for polygon poly (N,2).
#     area >=0.
#     """
#     x = poly[:, 0]
#     y = poly[:, 1]
#     x2 = torch.roll(x, shifts=-1, dims=0)
#     y2 = torch.roll(y, shifts=-1, dims=0)
#     cross = x * y2 - x2 * y
#     area2 = cross.sum()
#     area = 0.5 * area2.abs()

#     # Centroid formula; handle near-zero area gracefully
#     cx = ((x + x2) * cross).sum()
#     cy = ((y + y2) * cross).sum()
#     denom = (3.0 * area2).abs().clamp_min(1e-18)
#     c = torch.stack([cx, cy]) / denom
#     return area, c


# def _convex_hull(points: torch.Tensor) -> torch.Tensor:
#     """
#     Monotonic chain convex hull. points: (N,2).
#     Returns hull vertices in CCW order without repeating last point.
#     """
#     pts = points.detach().cpu().numpy()
#     pts = np.unique(pts, axis=0)
#     if len(pts) <= 2:
#         return _to_torch_xy(pts)

#     pts = sorted(pts.tolist())

#     def cross(o, a, b):
#         return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

#     lower = []
#     for p in pts:
#         while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
#             lower.pop()
#         lower.append(p)

#     upper = []
#     for p in reversed(pts):
#         while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
#             upper.pop()
#         upper.append(p)

#     hull = lower[:-1] + upper[:-1]
#     return _to_torch_xy(np.array(hull, dtype=np.float64))


# def _polygon_distance(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
#     """
#     Exact distance between two polygons via segment-point distances if disjoint; 0 if intersect/contain.
#     poly1, poly2: (N,2), (M,2)
#     """
#     if _polygons_intersect(poly1, poly2):
#         return torch.tensor(0.0, dtype=torch.float64)

#     a1, a2 = _polygon_edges(poly1)
#     b1, b2 = _polygon_edges(poly2)

#     # min distance among:
#     # - vertices of poly1 to edges of poly2
#     # - vertices of poly2 to edges of poly1
#     min_d = torch.tensor(float("inf"), dtype=torch.float64)

#     # v in poly1 to edges in poly2
#     for v in poly1:
#         for e1, e2 in zip(b1, b2):
#             d = _segment_distance_point(v, e1, e2)
#             if d < min_d:
#                 min_d = d

#     # v in poly2 to edges in poly1
#     for v in poly2:
#         for e1, e2 in zip(a1, a2):
#             d = _segment_distance_point(v, e1, e2)
#             if d < min_d:
#                 min_d = d

#     return min_d


# def _polygons_intersect(poly1: torch.Tensor, poly2: torch.Tensor) -> bool:
#     """
#     Intersection test: edge intersection OR containment.
#     """
#     a1, a2 = _polygon_edges(poly1)
#     b1, b2 = _polygon_edges(poly2)

#     for e1s, e1e in zip(a1, a2):
#         for e2s, e2e in zip(b1, b2):
#             if _segments_intersect(e1s, e1e, e2s, e2e):
#                 return True

#     # containment
#     if _point_in_polygon(poly1[0], poly2):
#         return True
#     if _point_in_polygon(poly2[0], poly1):
#         return True

#     return False


# def _sat_penetration_depth(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
#     """
#     Approx penetration depth via SAT on convex hulls:
#     If polygons intersect, returns min overlap along candidate axes.
#     Else returns 0.

#     Works robustly for your convex-hull usage.
#     """
#     if not _polygons_intersect(poly1, poly2):
#         return torch.tensor(0.0, dtype=torch.float64)

#     h1 = _convex_hull(poly1)
#     h2 = _convex_hull(poly2)

#     def axes(h):
#         a, b = _polygon_edges(h)
#         e = b - a
#         # normals
#         n = torch.stack([-e[:, 1], e[:, 0]], dim=-1)
#         # normalize
#         norm = torch.linalg.norm(n, dim=-1).clamp_min(1e-18)
#         return n / norm[:, None]

#     all_axes = torch.cat([axes(h1), axes(h2)], dim=0)

#     min_overlap = torch.tensor(float("inf"), dtype=torch.float64)

#     for ax in all_axes:
#         p1 = _dot2(h1, ax)
#         p2 = _dot2(h2, ax)
#         min1, max1 = p1.min(), p1.max()
#         min2, max2 = p2.min(), p2.max()
#         overlap = torch.min(max1, max2) - torch.max(min1, min2)
#         # If overlap negative => should not intersect, but numerical
#         overlap = overlap.clamp_min(0.0)
#         if overlap < min_overlap:
#             min_overlap = overlap

#     if torch.isinf(min_overlap):
#         return torch.tensor(0.0, dtype=torch.float64)
#     return min_overlap


# def _signed_distance_point_to_polygon(point: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
#     """
#     Signed distance from point to polygon:
#     negative if inside, positive if outside; magnitude is distance to edges.
#     """
#     a, b = _polygon_edges(poly)
#     dmin = torch.tensor(float("inf"), dtype=torch.float64)
#     for e1, e2 in zip(a, b):
#         d = _segment_distance_point(point, e1, e2)
#         if d < dmin:
#             dmin = d
#     inside = _point_in_polygon(point, poly)
#     return -dmin if inside else dmin


# # =========================
# # Minimal shapely-like shim
# # =========================

# class _Exterior:
#     def __init__(self, poly: "TorchPolygon"):
#         self._poly = poly

#     @property
#     def coords(self):
#         v = self._poly.vertices.detach().cpu().numpy()
#         # shapely repeats the first point at the end for exterior coords
#         if len(v) > 0:
#             v = np.vstack([v, v[0]])
#         return v

#     def distance(self, point: "TorchPoint") -> float:
#         p = _to_torch_xy(point.coords)[0]
#         poly = self._poly.vertices
#         a, b = _polygon_edges(poly)
#         dmin = float("inf")
#         for e1, e2 in zip(a, b):
#             d = float(_segment_distance_point(p, e1, e2))
#             dmin = min(dmin, d)
#         return dmin


# class _Centroid:
#     def __init__(self, poly: "TorchPolygon"):
#         self._poly = poly

#     @property
#     def coords(self):
#         _, c = _polygon_area_and_centroid(self._poly.vertices)
#         return np.array([[float(c[0]), float(c[1])]], dtype=np.float64)


# class TorchPoint:
#     def __init__(self, xy):
#         xy = _to_torch_xy(xy)
#         assert xy.shape[0] == 1
#         self._xy = xy

#     @property
#     def coords(self):
#         return self._xy.detach().cpu().numpy()[0]


# class TorchPolygon:
#     def __init__(self, vertices):
#         v = _to_torch_xy(vertices)
#         assert v.shape[0] >= 3, "Polygon needs at least 3 vertices"
#         self.vertices = v

#     @property
#     def convex_hull(self) -> "TorchPolygon":
#         return TorchPolygon(_convex_hull(self.vertices))

#     @property
#     def exterior(self) -> _Exterior:
#         return _Exterior(self)

#     @property
#     def centroid(self) -> _Centroid:
#         return _Centroid(self)

#     @property
#     def area(self) -> float:
#         area, _ = _polygon_area_and_centroid(self.vertices)
#         return float(area)

#     def distance(self, other) -> float:
#         if isinstance(other, TorchPolygon):
#             return float(_polygon_distance(self.vertices, other.vertices))
#         if isinstance(other, TorchPoint):
#             p = _to_torch_xy(other.coords)[0]
#             # shapely polygon.distance(point) returns 0 if inside
#             if _point_in_polygon(p, self.vertices):
#                 return 0.0
#             return float(_signed_distance_point_to_polygon(p, self.vertices))
#         raise TypeError(f"Unsupported distance to {type(other)}")

#     def contains(self, point: TorchPoint) -> bool:
#         p = _to_torch_xy(point.coords)[0]
#         return _point_in_polygon(p, self.vertices)

#     def buffer(self, r: float, k: int = 64) -> "TorchPolygon":
#         """
#         Approx polygon offset by r using Minkowski sum with a k-gon circle, then convex hull.
#         Good enough for your Circle approximation use-case.
#         """
#         r = float(r)
#         angles = torch.linspace(0, 2 * torch.pi, steps=k, dtype=torch.float64)[:-1]
#         circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * r  # (k,2)
#         # Minkowski sum samples
#         pts = (self.vertices[:, None, :] + circle[None, :, :]).reshape(-1, 2)
#         hull = _convex_hull(pts)
#         return TorchPolygon(hull)

#     def _bbox_center(self) -> torch.Tensor:
#         v = self.vertices
#         mn = v.min(dim=0).values
#         mx = v.max(dim=0).values
#         return (mn + mx) / 2.0


# class sh:  # noqa: N801 (keep name for test compatibility)
#     Point = TorchPoint
#     Polygon = TorchPolygon


# class af:  # noqa: N801 (keep name for test compatibility)
#     @staticmethod
#     def translate(poly: TorchPolygon, xoff: float, yoff: float) -> TorchPolygon:
#         t = torch.tensor([float(xoff), float(yoff)], dtype=torch.float64)
#         return TorchPolygon(poly.vertices + t)

#     @staticmethod
#     def rotate(poly: TorchPolygon, angle: float, origin="center", use_radians: bool = False) -> TorchPolygon:
#         theta = float(angle)
#         if not use_radians:
#             theta = theta * np.pi / 180.0

#         c = poly._bbox_center() if origin == "center" else poly._bbox_center()
#         v = poly.vertices - c[None, :]
#         ct = float(np.cos(theta))
#         st = float(np.sin(theta))
#         R = torch.tensor([[ct, -st], [st, ct]], dtype=torch.float64)
#         vr = v @ R.T
#         return TorchPolygon(vr + c[None, :])


# # =========================
# # Original code (API preserved)
# # =========================

# class IColor(Enum):
#     N = 0
#     R = 1
#     G = 2
#     B = 3


# class SpatialInterface(ABC):
#     """
#     Interface for spatial relation logic. All objects need to provide a quantitative semantic.
#     """

#     @abstractmethod
#     def shapes(self) -> set:
#         pass

#     @abstractmethod
#     def distance(self, other: 'SpatialInterface') -> float:
#         pass

#     @abstractmethod
#     def overlap(self, other: 'SpatialInterface') -> float:
#         pass

#     @abstractmethod
#     def enclosed_in(self, other: 'SpatialInterface') -> float:
#         pass

#     @abstractmethod
#     def proximity(self, other: 'SpatialInterface', eps: float) -> bool:
#         pass

#     @abstractmethod
#     def distance_compare(self, other: 'SpatialInterface', eps: float, fun) -> bool:
#         pass

#     @abstractmethod
#     def touching(self, other: 'SpatialInterface') -> bool:
#         pass

#     @abstractmethod
#     def angle(self, other: 'SpatialInterface') -> bool:
#         pass

#     @abstractmethod
#     def above(self, other: 'SpatialInterface') -> bool:
#         pass

#     @abstractmethod
#     def below(self, other: 'SpatialInterface') -> bool:
#         pass

#     @abstractmethod
#     def left_of(self, other: 'SpatialInterface') -> bool:
#         pass

#     @abstractmethod
#     def right_of(self, other: 'SpatialInterface') -> bool:
#         pass

#     @abstractmethod
#     def close_to(self, other: 'SpatialInterface') -> bool:
#         pass

#     @abstractmethod
#     def far_from(self, other: 'SpatialInterface') -> bool:
#         pass

#     @abstractmethod
#     def closer_to_than(self, closer: 'SpatialInterface', than: 'SpatialInterface') -> bool:
#         pass

#     @abstractmethod
#     def enlarge(self, radius: float) -> 'SpatialInterface':
#         pass

#     @abstractmethod
#     def __or__(self, other: 'SpatialInterface'):
#         pass

#     @abstractmethod
#     def __sub__(self, other: 'SpatialInterface'):
#         pass


# class ObjectInTime(ABC):
#     @abstractmethod
#     def getObject(self, time) -> 'SpatialInterface':
#         pass

#     @abstractmethod
#     def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
#         pass


# class DynamicObject(ObjectInTime):
#     def __init__(self):
#         self._shapes = OrderedDict()
#         self._latest_time = None

#     def addObject(self, object: SpatialInterface, time: int):
#         if self._latest_time is None:
#             self._latest_time = time - 1
#         assert time not in self._shapes, '<DynamicObject/add>: time step already added! t={}'.format(time)
#         assert time == self._latest_time + 1, '<DynamicObject/add>: time step missing! t = {}'.format(time)
#         self._shapes[time] = object
#         self._latest_time = time

#     def getObject(self, time) -> 'SpatialInterface':
#         assert time in self._shapes, '<DynamicObject/add>: time step not yet added! t={}'.format(time)
#         return self._shapes[time]

#     def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
#         assert idx < len(self._shapes) if idx >= 0 else abs(idx) <= len(self._shapes)
#         return list(self._shapes.values())[idx]

#     def __or__(self, other):
#         if isinstance(other, (StaticObject, DynamicObject)):
#             return ObjectCollection(self, other)
#         elif isinstance(other, ObjectCollection):
#             return other + self
#         else:
#             raise Exception('<DynamicObject/add>: Provided object not supported! other = {}'.format(other))


# class StaticObject(ObjectInTime):
#     def __init__(self, spatial_object: SpatialInterface):
#         super().__init__()
#         self._spatial_obj = spatial_object

#     def getObject(self, time) -> 'SpatialInterface':
#         return self._spatial_obj

#     def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
#         return self._spatial_obj

#     def __or__(self, other):
#         if isinstance(other, (StaticObject, DynamicObject)):
#             return ObjectCollection(self, other)
#         elif isinstance(other, ObjectCollection):
#             return other + self
#         else:
#             raise Exception('<DynamicObject/add>: Provided object not supported! other = {}'.format(other))


# class ObjectCollection(ObjectInTime):
#     def __init__(self, *args):
#         self._object_set = set(args)

#     def getObject(self, time) -> 'SpatialInterface':
#         objs = [o.getObject(time) for o in self._object_set]
#         shapes = [o.shapes() for o in objs]
#         shapes = shapes[0].union(*shapes[1:])
#         assert len(objs) > 0
#         return type(objs[0])(shapes)

#     def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
#         objs = [o.getObjectByIndex(idx) for o in self._object_set]
#         shapes = [o.shapes for o in objs]  # keep original bug/behavior
#         shapes = shapes[0].union(*shapes[1:])
#         assert len(objs) > 0
#         return type(objs[0])(shapes)

#     def __len__(self):
#         return len(self._object_set)

#     @property
#     def objects(self):
#         return self._object_set

#     def __or__(self, other):
#         collection = ObjectCollection()
#         if isinstance(other, ObjectCollection):
#             collection._object_set = self._object_set | other._object_set
#         elif isinstance(other, (StaticObject, DynamicObject)):
#             collection._object_set = self._object_set | {other}
#         else:
#             raise Exception('<ObjectCollection/add>: Provided object not supported! other = {}'.format(other))
#         return collection

#     def __sub__(self, other):
#         collection = ObjectCollection()
#         if isinstance(other, ObjectCollection):
#             collection._object_set = self._object_set - other._object_set
#         elif isinstance(other, (StaticObject, DynamicObject)):
#             collection._object_set = self._object_set - {other}
#         else:
#             raise Exception('<ObjectCollection/add>: Provided object not supported! other = {}'.format(other))
#         return collection

#     def __and__(self, other):
#         collection = ObjectCollection()
#         if isinstance(other, ObjectCollection):
#             collection._object_set = self._object_set & other._object_set
#         elif isinstance(other, (StaticObject, DynamicObject)):
#             collection._object_set = self._object_set & {other}
#         else:
#             raise Exception('<ObjectCollection/add>: Provided object not supported! other = {}'.format(other))
#         return collection


# class Polygon(object):
#     """
#     Class representing a polygon
#     """
#     _id = 0
#     _ORIGIN = sh.Point([0, 0])

#     @classmethod
#     def _get_id(cls):
#         cls._id += 1
#         return cls._id

#     def __init__(self, vertices: np.ndarray, color: IColor = IColor.N, convex_hull: bool = True):
#         assert isinstance(vertices, np.ndarray), '<Polygon/init>: vertices must be of type np.ndarray!'
#         if convex_hull:
#             self.shape = sh.Polygon(vertices).convex_hull
#         else:
#             self.shape = sh.Polygon(vertices)
#         self.color = color
#         self.id = self._get_id()

#     @property
#     def shape(self) -> TorchPolygon:
#         return self._shape

#     @shape.setter
#     def shape(self, shape: TorchPolygon):
#         assert isinstance(shape, TorchPolygon), '<Polygon/shape>: Only polygon shim is supported'
#         self._shape = shape

#     @property
#     def vertices(self) -> np.ndarray:
#         return np.array(self.shape.exterior.coords)

#     @property
#     def center(self) -> np.ndarray:
#         return np.array(self.shape.centroid.coords[0])

#     def enlarge(self, radius: float) -> 'Polygon':
#         enlarged = self.shape.buffer(radius)
#         return Polygon(np.array(enlarged.exterior.coords))

#     def translate(self, t: np.ndarray):
#         assert len(t) == 2
#         self.shape = af.translate(self.shape, t[0], t[1])
#         return self

#     def rotate(self, theta: float, from_origin: bool = True, use_radians=False):
#         self.shape = af.rotate(self.shape, theta, origin='center', use_radians=use_radians)
#         return self

#     def distance(self, other: 'Polygon'):
#         return self.shape.distance(other.shape)

#     def penetration_depth(self, other: 'Polygon'):
#         v1 = _to_torch_xy(self.shape.exterior.coords[:-1])
#         v2 = _to_torch_xy(other.shape.exterior.coords[:-1])
#         return float(_sat_penetration_depth(v1, v2))

#     def signed_distance(self, other: 'Polygon'):
#         gjk = self.distance(other)
#         if gjk <= 0.0000001:
#             pd = self.penetration_depth(other)
#             return gjk - pd
#         return gjk

#     def enclosedIn(self, other: 'Polygon'):
#         # >=0 if enclosed, <0 otherwise
#         poly_other = _to_torch_xy(other.shape.exterior.coords[:-1])
#         worst = torch.tensor(-float("inf"), dtype=torch.float64)
#         for v in self.vertices[:-1]:
#             p = _to_torch_xy(v)[0]
#             sd = _signed_distance_point_to_polygon(p, poly_other)
#             if sd > worst:
#                 worst = sd
#         # If worst <= 0 => all inside => return -worst (>=0), else negative
#         val = -worst
#         return float(val) if not torch.isclose(worst, torch.tensor(0.0, dtype=torch.float64)) else float(worst)

#     def contains_point(self, point: np.ndarray):
#         return self.shape.contains(sh.Point(point))

#     @property
#     def color(self) -> IColor:
#         return self._color

#     @color.setter
#     def color(self, color: IColor):
#         self._color = color

#     def plot(self, ax=None, alpha=1.0, label: bool = True, color='k'):
#         if ax is None:
#             ax = plt.gca()

#         ax.add_patch(mp.Polygon(self.vertices, color=color, alpha=alpha))
#         if label:
#             plt.text(self.center[0], self.center[1], s=str(self.id), c='white',
#                      bbox=dict(facecolor='white', alpha=0.5))

#     def minkowski_sum(self, other: 'Polygon', sub: bool = False) -> 'Polygon':
#         new_vertices = list()
#         for v in other.vertices:
#             if not sub:
#                 new_vertices.append(self.vertices + (v - other.center))
#             else:
#                 new_vertices.append(self.vertices - (v - other.center))
#         return Polygon(np.vstack(new_vertices))

#     def __add__(self, other):
#         return self.minkowski_sum(other)

#     def __sub__(self, other):
#         return self.minkowski_sum(other, sub=True)

#     def __hash__(self):
#         return self.id


# class Circle(Polygon):
#     def __init__(self, center: np.ndarray, r: float):
#         vertices = np.array(sh.Point(center).buffer(r).exterior.coords)
#         super().__init__(vertices, convex_hull=False)


# class PolygonCollection(SpatialInterface):
#     """
#     Implements spatial interface for objects of type polytope. Represents set of polytopes
#     """

#     def __init__(self, polygons: Set[Polygon]):
#         self.polygons = polygons if isinstance(polygons, set) else set(polygons)

#     @property
#     def polygons(self) -> Set[Polygon]:
#         return self._polygons

#     @polygons.setter
#     def polygons(self, polygons: Set[Polygon]):
#         self._polygons = polygons

#     def add(self, p: Polygon):
#         self.polygons.add(p)

#     def remove(self, p: Polygon):
#         self.polygons.discard(p)

#     def shapes(self) -> set:
#         return self.polygons

#     def of_color(self, color: IColor) -> 'PolygonCollection':
#         return PolygonCollection(set([p for p in self.polygons if p.color == color]))

#     def plot(self, ax=None, color='k', label=True):
#         if ax is None:
#             ax = plt.gca()
#         for p in self.polygons:
#             p.plot(ax=ax, label=label, color=color)
#         plt.autoscale()
#         plt.axis('equal')

#     def distance(self, other: 'SpatialInterface') -> float:
#         assert isinstance(other, PolygonCollection), \
#             '<Polygon/distance>: Other object must be of type polygon, got {}'.format(other)

#         result = list()
#         for p in self.polygons:
#             result.append([p.signed_distance(o) for o in other.polygons])
#         return result

#     def overlap(self, other: 'SpatialInterface') -> bool:
#         inter = list()
#         for p in self.polygons:
#             inter.append([-p.signed_distance(o) for o in other.polygons])
#         inter = np.array(inter)
#         return np.max(inter)

#     def enclosed_in(self, other: 'SpatialInterface') -> float:
#         enclosed = list()
#         for p in self.polygons:
#             enclosed.append(np.array([p.enclosedIn(o) for o in other.polygons]).max())
#         return np.array(enclosed).min()

#     def proximity(self, other: 'SpatialInterface', eps: float) -> bool:
#         return self.distance_compare(other, eps, np.less_equal)

#     def distance_compare(self, other: 'SpatialInterface', eps: float, fun):
#         assert eps > 0, '<Polygon>: Epsilon must be positive, got {}'.format(eps)

#         if fun == np.less_equal:
#             return np.max(np.repeat(eps, len(other.polygons)) - self.distance(other))
#         if fun == np.greater_equal:
#             return np.max(self.distance(other) - np.repeat(eps, len(other.polygons)))
#         if fun == np.equal:
#             return np.min([
#                 np.max(np.repeat(eps, len(other.polygons)) - self.distance(other)),
#                 np.max(self.distance(other) - np.repeat(eps, len(other.polygons)))
#             ])

#     def touching(self, other: 'SpatialInterface', eps: float = 5) -> bool:
#         return self.proximity(other, eps=eps)
#         return np.min([self.proximity(other, eps=eps), -self.proximity(other, eps=-eps)])

#     def _min(self, axis: int) -> float:
#         return np.min([c.center[axis] for c in self.polygons])

#     def _max(self, axis: int) -> float:
#         return np.max([c.center[axis] for c in self.polygons])

#     def left_of(self, other: 'SpatialInterface') -> float:
#         return other._min(0) - self._max(0)

#     def right_of(self, other: 'SpatialInterface') -> float:
#         return self._min(0) - other._max(0)

#     def above(self, other: 'SpatialInterface') -> float:
#         return self._min(1) - other._max(1)

#     def below(self, other: 'SpatialInterface') -> float:
#         return other._min(1) - self._max(1)

#     def close_to(self, other: 'SpatialInterface') -> float:
#         return self.proximity(other, 70.)

#     def far_from(self, other: 'SpatialInterface') -> float:
#         return -self.proximity(other, 150)

#     def closer_to_than(self, closer: 'SpatialInterface', than: 'SpatialInterface') -> float:
#         return np.min(self.distance(than)) - np.min(self.distance(closer))

#     def enlarge(self, radius: float) -> 'SpatialInterface':
#         return PolygonCollection(set([p.enlarge(radius) for p in self.polygons]))

#     def angle(self, other: 'CircleOLD') -> float:
#         pass

#     def __or__(self, other: 'PolytopeCollection'):
#         return PolygonCollection(self.polygons | other.polygons)

#     def __sub__(self, other: 'PolytopeCollection'):
#         return PolygonCollection(self.polygons - other.polygons)


# if __name__ == '__main__':

#     p1 = Polygon(np.array([[0, 0], [3, 3], [6, 0]]))
#     p2 = Polygon(np.array([[3, 5], [7, 8], [10, 6]]))
#     p2 = p2.rotate(30.45)
#     p3 = Polygon(np.array([[3, 5], [7, 8], [10, 6]]) - 4, IColor.B)

#     p_sum = p1.minkowski_sum(p2)
#     (p1 + p2).plot(color='r')
#     (p1 - p2).plot(color='g')
#     plt.autoscale()
#     plt.show()
#     print(p1)

#     p1.plot()
#     p2.plot()
#     p3.plot()
#     plt.autoscale()
#     plt.show()
#     import time

#     a = sh.Polygon(p1.vertices)
#     b = sh.Polygon(p2.vertices)
#     print('Area is {}'.format(a.area))
#     print('Distance is {}'.format(a.distance(b)))

#     t0 = time.time()
#     for i in range(100):
#         a.distance(b)
#     print(f'Time took {time.time() - t0}')

#     pc = PolygonCollection(set([p1, p2]))
#     pd = PolygonCollection(set([p3]))

#     t0 = time.time()
#     print('Distance is {}'.format(pc.distance(pd)))
#     print('Distance is {}'.format(pc.distance(pc)))
#     print('Intersecting is {}'.format(pc.overlap(pd)))
#     print('Intersecting is {}'.format(pc.overlap(pc)))
#     print(f'Time took {time.time() - t0}')

#     t0 = time.time()
#     for i in range(1):
#         pc.distance(pc)
#     print(f'Time took for 10 {time.time() - t0}')
