# What is Quadrants Typing?

A typing layer for the
[quadrants](https://github.com/Genesis-Embodied-AI/quadrants), plus PEP 561 type
stubs for the `quadrants` package itself. Installing this one package gives
quadrants users both:

- **`quadrants_typing`** — generic `Vec`, `Mat`, `NDArray`, dimension, and dtype
  types for annotating quadrants kernels. Import it under the conventional
  alias:

  ```python
  import quadrants_typing as qdt
  import quadrants as qd
  
  @qd.kernel
  def add(
      x: qdt.NDArray[qd.i32, qdt.Dim1], 
      y: qdt.NDArray[qd.i32, qdt.Dim1],
  ) -> None:
      for i in range(x.shape[0]):
          x[i] = y[i]
  ```

- **Automatic `quadrants` stubs** — type checkers (pyright, mypy) discover the
  bundled `quadrants-stubs` package via PEP 561 with no configuration, so
  `import quadrants` is fully typed.

## Quadrants Replacements

Type annotations replaces *directly* some quadrants functions, mostly from
`qd.types`. Here is a full table:

* `N`,`R`,`C` in `{1,2,3,4,5,6,7,8}`
* `dtype` n `{qd.u8, qd.u16, qd.u32, ... etc.}` 

| Description | `qdt` version | `qd` version | Example |
|:-:|:-:|:-:|:-:|
| Concrete tensor type | `qdt.NDArray[qd.<dtype>, Dim<N>]` | `qd.types.NDArray[qd.<dtype>, <N>]` | `qdt.NDArray[qd.i32, Dim3]` | 
| Templated tensor type | `qdt.NDArray[T, DimAny]` | `qd.types.ndarray(dtype=None, ndim=None)` | `def f[T: (qd.i32, qd.f32)](arr: qdt.NDArray[T, DimAny])` |
| Vector type | `qdt.Vec[qd.<dtype>, Dim<N>]` | `qd.types.vector(<N>, qd.<dtype>)` | `qdt.Vec[qd.f32, Dim3]` |
| Matrix type | `qdt.Mat[qd.<dtype>, Dim<R>, Dim<C>]` | `qd.types.matrix(<R>, <C>, qd.<dtype>)` | `qdt.Mat[qd.f32, Dim3, Dim4]` |
| Compile-time value | `qdt.Template[T]` | `qd.template()` | `qdt.Template[bool]` |

If you would like to construct tensors with a rank greater than 8, or vectors
and matrices with any dimension longer than 8, you can. See `weird_hacks`
example for how to create the annotation, or use `qd.types.vector` or
`qd.types.matrix` if you only need a constructor (e.g. for a local array).

## Examples

Runnable, annotated example programs live in `examples/`. Run one by name:

```sh
pixi run example init_ndarray
```

## How Finished is This?

Not at all! There's lots of work still to be done :P. Please help!
