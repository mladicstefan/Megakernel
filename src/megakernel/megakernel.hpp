#pragma once
#include "config/config.hpp"
#include "reference/forward.hpp"

#include <cstdint>
#include <memory>
#include <variant>

// The megakernel: one kernel launch runs a whole decode step. Every block of the grid walks
// the same list of phases, and a grid-wide barrier separates a phase from the one that reads
// its results. nvcc compiles this header: C++20 only.
namespace mk {
// weights/verified_device.hpp, host only, defines it: make() needs only that one exists.
template <GpuSpec G>
struct VerifiedDevice;

// What the blocks of one launch share, on the device.
template <MegakernelConfig C>
struct MegaSync {
    unsigned arrived; // blocks that have reached a barrier, counted over every launch
    unsigned failed;  // set when a barrier gave up waiting
    unsigned token;   // the step's most likely next token
    // Each block's best logit and its token, before the last reduction.
    float    best_logit[Grid<C>::blocks];
    unsigned best_token[Grid<C>::blocks];
};

// The megakernel for configuration C. Its grid is Grid<C>: every SM of C::Gpu runs
// C::Gpu::blocks_per_sm blocks, all resident at once, which the barriers rely on. It decodes
// into a Decoder<C>, which owns the KV cache and the trace, and writes the tensors it
// materializes into that trace: embd, Qcur, Kcur, Vcur, kqv_out, ffn_inp, ffn_swiglu, l_out
// and result_output. The other fields of the trace are fused away and stay untouched.
template <MegakernelConfig C>
struct Megakernel {
    std::unique_ptr<MegaSync<C>, CudaFree> sync;
    std::unique_ptr<void, CudaFree>        partials; // each block's part of an attention softmax
    std::uint32_t                          epoch = 0; // what `arrived` was before the next launch
    std::unique_ptr<Trace, CudaFree>       operands;  // for decode_from; null until it is used

    // Only on a device verified to be C::Gpu. Fails if the device cannot hold the whole grid
    // at once, or if the kernel's shared memory is not what the capacity proof counted.
    static std::variant<Megakernel, CudaError> make(const VerifiedDevice<typename C::Gpu>&);

    // Runs token at position d.n_past, which must be below C::n_ctx, and returns the most
    // likely next token.
    std::variant<Token, CudaError> decode(Decoder<C>& d, Token token);

    // The same step, with every phase taking its operands from another implementation's
    // trace of this position. See Decoder::decode_from.
    std::variant<Token, CudaError> decode_from(Decoder<C>& d, Token token, const Trace& from);
};

// megakernel.cu defines the members for this configuration.
extern template struct Megakernel<EngineConfig>;
} // namespace mk
