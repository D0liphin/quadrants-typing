import quadrants as qd

from quadrants_typing._dim import Dim2, Dim3, Dim4
from quadrants_typing._mat import Mat
from quadrants_typing._vec import Vec

Vec2f = Vec[qd.f32, Dim2]
Vec3f = Vec[qd.f32, Dim3]
Vec4f = Vec[qd.f32, Dim4]
Vec2i = Vec[qd.i32, Dim2]
Vec3i = Vec[qd.i32, Dim3]
Vec4i = Vec[qd.i32, Dim4]

Mat2f = Mat[qd.f32, Dim2, Dim2]
Mat3f = Mat[qd.f32, Dim3, Dim3]
Mat4f = Mat[qd.f32, Dim4, Dim4]
Mat2i = Mat[qd.i32, Dim2, Dim2]
Mat3i = Mat[qd.i32, Dim3, Dim3]
Mat4i = Mat[qd.i32, Dim4, Dim4]
