# Compile time scheduled decode megakernel
```cpp
struct ModelCfg { int hidden, inter, layers, heads, kv_heads, head_dim, vocab; };
inline constexpr ModelCfg kSmolLM2{576, 1536, 30, 9, 3, 64, 49152};
inline constexpr ModelCfg kLlama1B{2048, 8192, 16, 32, 8, 64, 128256};

struct GpuSpec { int sms; };
inline constexpr GpuSpec k4070TiS{66}, k5070{48};

template <ModelCfg M, GpuSpec G, class Fmt>
struct Plan {
  static constexpr auto graph = build_layer_graph<M, Fmt>();  // tiles + edges
  static constexpr auto sched = list_schedule(graph, G.sms);  // per-SM order
  static_assert(acyclic(graph, sched), "schedule can deadlock");
  static_assert(disjoint_writes(graph, sched), "schedule has a write race");
};

template <ModelCfg M, GpuSpec G, class Fmt>
__global__ void __launch_bounds__(256, 1) decode_step(Params p) {
  run_my_tasks<Plan<M, G, Fmt>>(blockIdx.x, p);  // mode A or mode B
}
```
