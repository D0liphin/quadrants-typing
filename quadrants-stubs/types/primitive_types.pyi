import quadrants_typing as qdt

# `i32`/`f32` collapse to the builtins `int`/`float`, which makes code much more
# readable. quadrants is configured with 32-bit default int/float here (hence the
# `_f32_i32` package name); rebuild with the other widths collapsed to retype.
i32 = int
f32 = float

# The remaining dtypes stay nominal so that same-width-only arithmetic among
# them (e.g. `f16 + f16`) is still enforced by `qdt.NumericOps[Self]`.
class f16(qdt.Floating, qdt.Number): ...
class f64(qdt.Floating, qdt.Number): ...
class i8(qdt.SignedInteger, qdt.Number): ...
class i16(qdt.SignedInteger, qdt.Number): ...
class i64(qdt.SignedInteger, qdt.Number): ...
class u1(qdt.UnsignedInteger, qdt.Number): ...
class u8(qdt.UnsignedInteger, qdt.Number): ...
class u16(qdt.UnsignedInteger, qdt.Number): ...
class u32(qdt.UnsignedInteger, qdt.Number): ...
class u64(qdt.UnsignedInteger, qdt.Number): ...

# Long-form dtype aliases. quadrants exposes both the terse (`i8`) and the
# NumPy-style verbose (`int8`) spelling of every primitive; they are the exact
# same object, so the aliases are plain assignments.
int8 = i8
int16 = i16
int32 = i32
int64 = i64
uint1 = u1
uint8 = u8
uint16 = u16
uint32 = u32
uint64 = u64
float16 = f16
float32 = f32
float64 = f64
