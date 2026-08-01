# `quadrants` 1.2.0 — public API surface & stub coverage

Legend:

- `.` after a name → it's a module/subpackage.
- `*` → the name is a **re-export** (its canonical definition lives in another
  public module listed here; e.g. top-level `cos*` is `quadrants.math.cos`).
- `✔` → covered by the stubs / typing layer.
- `x` → **not** covered (falls through `__getattr__` → `Any`, i.e. untyped).
- `?` → **partial** (module/name is only partly stubbed).

Coverage is measured against `quadrants-stubs/` + `quadrants_typing/` as they
stand today. Anything reachable only via a top-level `__getattr__(name) -> Any`
is counted as `x` (no real types), even though it won't error.

---

```
quadrants.                                          ?

  # ── submodules ─────────────────────────────────────────────
  math.                                             ✔   (fully stubbed)
    pi ✔   e ✔   inf ✔   nan ✔
    # vec / mat constructors
    vec2 ✔   vec3 ✔   vec4 ✔
    ivec2 ✔  ivec3 ✔  ivec4 ✔
    uvec2 ✔  uvec3 ✔  uvec4 ✔
    mat2 ✔   mat3 ✔   mat4 ✔
    # unary element-wise
    acos ✔   asin ✔   cos ✔    sin ✔    tan ✔    tanh ✔
    exp ✔    log ✔    log2 ✔   sqrt ✔   fract ✔  sign ✔
    degrees ✔  radians ✔
    ceil ✔   floor ✔  round ✔
    # binary / ternary element-wise
    atan2 ✔  mod ✔    pow ✔    clamp ✔  mix ✔    smoothstep ✔  step ✔
    max ✔    min ✔
    # predicates / integer ops
    isinf ✔  isnan ✔  popcnt ✔  clz ✔   ffs ✔    fns ✔
    # vector ops
    dot ✔    cross ✔  normalize ✔  reflect ✔  refract ✔  length ✔  distance ✔
    # matrix ops
    determinant ✔  inverse ✔  eye ✔
    # complex vec2 ops
    cconj ✔  cdiv ✔   cexp ✔   cinv ✔   clog ✔   cmul ✔   csqrt ✔  cpow ✔
    # transforms
    vdir ✔   rotation2d ✔  rotation3d ✔  rot_by_axis ✔  rot_yaw_pitch_roll ✔
    scale ✔  translate ✔

  types.                                            ?
    # primitive dtypes (stubbed)
    i8 ✔   i16 ✔   i32 ✔   i64 ✔
    u1 ✔   u8 ✔    u16 ✔   u32 ✔   u64 ✔
    f16 ✔  f32 ✔   f64 ✔
    # long-name dtype aliases (int8/float32/uint1/…)
    int8 ✔   int16 ✔   int32 ✔   int64 ✔
    uint1 ✔  uint8 ✔   uint16 ✔  uint32 ✔  uint64 ✔
    float16 ✔  float32 ✔  float64 ✔
    # compound-type factories
    vector ✔    matrix ✔    struct x (omitted: use dataclass)    ref x (todo)
    # dtype predicates
    is_integral ✔   is_real ✔   is_signed ✔   is_tensor ✔
    # annotation types
    NDArray x   Template x   template x   sparse_matrix_builder x
    BufferViewType x   ndarray x
    quant.                                          x (todo)
      int x   float x   fixed x

  simt.                                             ?
    block.                                          ✔
      SharedArray ✔   sync ✔   mem_fence ✔   mem_sync ✔ (deprecated)
      thread_idx ✔    global_thread_idx ✔
      reduce ✔  reduce_add ✔  reduce_max ✔  reduce_min ✔
      reduce_all ✔  reduce_all_add ✔  reduce_all_max ✔  reduce_all_min ✔
      inclusive_add ✔  inclusive_max ✔  inclusive_min ✔  inclusive_scan ✔
      exclusive_add ✔  exclusive_max ✔  exclusive_min ✔  exclusive_scan ✔
      sync_all_nonzero ✔  sync_any_nonzero ✔  sync_count_nonzero ✔
      arch_uses_spv ✔  radix_rank_match_atomic_or ✔
    warp.                                           ✔
      active_mask ✔  all_nonzero ✔  any_nonzero ✔  ballot ✔  unique ✔  sync ✔
      match_all ✔  match_any ✔
      shfl_sync_i32 ✔  shfl_sync_f32 ✔  shfl_up_i32 ✔  shfl_up_f32 ✔
      shfl_down_i32 ✔  shfl_down_f32 ✔  shfl_xor_i32 ✔
    subgroup.                                       ✔
      # membership / election
      elect ✔  barrier ✔  group_size ✔  log2_group_size ✔  invocation_id ✔
      all_true ✔  any_true ✔  all_equal ✔  ballot ✔  broadcast ✔  broadcast_first ✔
      ballot_first_n ✔  ballot_full_subgroup ✔
      all_true_tiled ✔  any_true_tiled ✔  all_equal_tiled ✔
      # lane masks
      lanemask_eq ✔  lanemask_lt ✔  lanemask_le ✔  lanemask_gt ✔  lanemask_ge ✔
      # shuffles
      shuffle ✔  shuffle_up ✔  shuffle_down ✔  shuffle_xor ✔
      mem_fence ✔  memory_barrier ✔  sync ✔
      # (+ reductions/sorting re-exports *)
    reductions.                                     ✔
      reduce_add ✔  reduce_max ✔  reduce_min ✔
      reduce_all_add ✔  reduce_all_max ✔  reduce_all_min ✔
      inclusive_add ✔  inclusive_max ✔  inclusive_min ✔  inclusive_mul ✔
      inclusive_and ✔  inclusive_or ✔  inclusive_xor ✔
      exclusive_add ✔  exclusive_max ✔  exclusive_min ✔  exclusive_mul ✔
      exclusive_and ✔  exclusive_or ✔  exclusive_xor ✔
      segmented_reduce_add ✔  segmented_reduce_max ✔  segmented_reduce_min ✔
      # (+ *_tiled variant of each of the above ✔)
    sorting.                                        ✔
      bitonic_sort_kv ✔   bitonic_sort_kv_tiled ✔
    grid.                                           x
      mem_fence x   memfence x   arch_uses_spv x
    tile_slicing.                                   x
      try_tile_ref x   try_tile_slice x

  ad.                                               x
    Tape x   FwdMode x   no_grad x   grad_for x   grad_replaced x
    clear_all_gradients x

  algorithms.                                       x
    PrefixSumExecutor x   parallel_sort x   sort x*   select x*
    reduce_add x*   reduce_max x*   reduce_min x*
    exclusive_scan_add x*   exclusive_scan_max x*   exclusive_scan_min x*
    reduce_by_key_add x*
    sort_scratch_slots x   select_scratch_slots x   reduce_scratch_slots x
    exclusive_scan_scratch_slots x   reduce_by_key_scratch_slots x

  linalg.                                           x
    SparseCG x   SparseSolver x

  sparse.                                           x
    grid x   usage x

  experimental.                                     x
    real_func x

  tools.                                            x
    PLYWriter x
    np2ply.                                         x
      PLYWriter x
    vtk.                                            x
      write_vtk x
    write_vtk x*
    diagnose.                                       x

  interop.                                          x
    get_mps_command_queue x

  profiler.                                         x
    CuptiMetric x
    print_kernel_profiler_info x   query_kernel_profiler_info x
    clear_kernel_profiler_info x   collect_kernel_profiler_metrics x
    get_kernel_profiler_total_time x   set_kernel_profiler_metrics x
    set_kernel_profiler_toolkit x   get_predefined_cupti_metrics x
    print_memory_profiler_info x
    print_scoped_profiler_info x   clear_scoped_profiler_info x
    kernel_metrics. x   kernel_profiler. x   memory_profiler. x   scoped_profiler. x

  graph.                                            x
    parallel x   parallel_context x   do_while x

  # ── top-level names ────────────────────────────────────────
  # backends / arch
  Arch ✔          Backend ✔
  cpu ✔   cuda ✔   metal ✔   vulkan ✔   amdgpu ✔
  x64 ✔   x86_64 ✔   arm64 ✔   python ✔   gpu ✔
  extension x

  # kernels / functions / decorators
  kernel ✔   func ✔   data_oriented ✔
  pyfunc x   real_func x   pure x   perf_dispatch x

  # lifecycle
  init ✔   reset ✔   sync ✔

  # iteration
  ndrange ✔   grouped ✔   loop_config ✔

  # casts / bit ops
  cast ✔   bit_cast ✔   bit_shr ✔
  raw_div ✔   raw_mod ✔   select ✔   random ✔

  # atomics
  atomic_add ✔  atomic_sub ✔  atomic_mul ✔  atomic_max ✔  atomic_min ✔
  atomic_and ✔  atomic_or ✔   atomic_xor ✔  atomic_exchange ✔  atomic_cas ✔

  # element-wise math (re-exported from math.)
  abs ✔
  acos* ✔   asin* ✔   atan2* ✔   cos* ✔    sin* ✔    tan* ✔    tanh* ✔
  exp* ✔    log* ✔    sqrt* ✔    pow* ✔     ceil* ✔   floor* ✔  round* ✔
  max* ✔    min* ✔
  frexp x   rsqrt x   volatile_load x

  # primitive dtypes (re-exported from types.)
  i8* ✔   i16* ✔   i32* ✔   i64* ✔
  u1* ✔   u8* ✔    u16* ✔   u32* ✔   u64* ✔
  f16* ✔  f32* ✔   f64* ✔
  # long-name dtype aliases
  int8 ✔   int16 ✔   int32 ✔   int64 ✔
  uint1 ✔  uint8 ✔   uint16 ✔  uint32 ✔  uint64 ✔
  float16 ✔  float32 ✔  float64 ✔
  ref x

  # data containers / fields (all x)
  Field x   ScalarField x   MatrixField x   StructField x
  Ndarray x   ScalarNdarray x   MatrixNdarray x   VectorNdarray x
  Matrix x   Vector x   Struct x
  Tensor x   MatrixTensor x   VectorTensor x
  BitpackedFields x   BufferView x   FieldsBuilder x   SNode x
  Mesh x   MeshInstance x   Template x
  field x   ndarray ✔   tensor x   wrap x   dataclass x
  one x   zero x

  # snode / layout
  root x   axes x   activate x   deactivate x   deactivate_all_snodes x
  append x   is_active x   get_addr x   rescale_index x   length ✔
  no_activate x   assume_in_range x
  Format x   Layout x

  # streams / events / graph
  Stream x   Event x   create_stream x   create_event x   stream_parallel x
  GraphStatus x   graph_parallel x   graph_parallel_context x   graph_do_while x

  # static / compile-time
  static ✔   static_assert x   static_print x   stop_grad x   template ✔
  checkpoint x   dump_compile_config x   is_extension_enabled x

  # misc intrinsics
  clock_counter x   clock_freq_hz x   global_thread_idx x
  block_local x   cache_read_only x   mesh_local x   mesh_patch_idx x

  # linear algebra free functions (from _funcs, all x)
  svd x   sym_eig x   eig x   solve x   polar_decompose x   make_spd x   randn x

  # device capability / sparse
  DeviceCapability x   sparse_matrix_builder x

  # logging
  set_logging_level x   is_logging_effective x
  CRITICAL x   ERROR x   WARN x   INFO x   DEBUG x   TRACE x

  # exceptions (all x)
  QuadrantsSyntaxError x   QuadrantsTypeError x   QuadrantsRuntimeError x
  QuadrantsRuntimeTypeError x   QuadrantsCompilationError x
  QuadrantsAssertionError x   QuadrantsNameError x

  # ndrange index-name constants (all x)
  i x  j x  k x  l x  ij x  ik x  il x  jk x  jl x  kl x
  ijk x  ijl x  ikl x  jkl x  ijkl x

  __getattr__ ✔   # top-level Any fallback keeps unlisted names non-erroring
```

---

## Summary of what's missing (the `x` work list)

Ordered roughly by likely impact for downstream users:

1. **Field / Ndarray / Matrix data model** — `Field`, `ScalarField`,
   `MatrixField`, `StructField`, `Ndarray` & friends, `Matrix`, `Vector`,
   `Struct`, plus the constructors `field`, `struct`, `dataclass`. This is the
   core runtime data model and is entirely untyped.
2. **`types.` gaps** — `NDArray`, `ndarray`, `struct`, `ref`, the
   `is_integral/is_real/is_signed/is_tensor` predicates, the long-name dtype
   aliases (`int32`, `float32`, …), and `types.quant`.
3. **`ad.` (autodiff)** — `Tape`, `FwdMode`, `no_grad`, `grad_for`, … completely
   unstubbed.
4. **`simt.`** — only `block.reduce_add` / `block.thread_idx` exist; `warp`,
   `subgroup`, `reductions`, `sorting`, `grid`, `tile_slicing` and the rest of
   `block` are all untyped.
5. **`algorithms.` / `linalg.` / `sparse.`** — sort/scan/reduce family, sparse
   solvers, and sparse grids.
6. **Linear-algebra free functions** — `svd`, `sym_eig`, `eig`, `solve`,
   `polar_decompose`, `make_spd`, `randn`.
7. **Snode / layout / streams / graph / profiler / interop / tools** —
   lower-priority but all untyped.
8. **Exceptions & logging** — the `Quadrants*Error` hierarchy and log-level
   constants (cheap wins).
```
