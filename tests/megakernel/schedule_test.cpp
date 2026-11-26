#include "megakernel/schedule.hpp"

#include <cstdint>

#include <gtest/gtest.h>

namespace {
using mk::partitions;
using mk::PhaseList;
using mk::Rows;
using mk::rows_of;

// Qwen3-0.6B.
constexpr unsigned kEmbd      = 1024;
constexpr unsigned kFf        = 3072;
constexpr unsigned kHeads     = 16;
constexpr unsigned kKvHeads   = 8;
constexpr unsigned kHeadDim   = 128;
constexpr unsigned kVocab     = 151936;
constexpr unsigned kGroup     = kHeads / kKvHeads;   // query heads per KV head
constexpr unsigned kQ         = kHeads * kHeadDim;   // the height of q, the width of o
constexpr unsigned kKvPerPos  = kKvHeads * kHeadDim; // the height of k and v
constexpr unsigned kHeights[] = {kQ, kKvPerPos, kEmbd, kFf, kVocab}; // of every matrix

// The engine's grid, 2 blocks of 256 threads on each of 66 SMs, and the same on 48 SMs.
constexpr unsigned kThreads  = 256;
constexpr unsigned kBlocks   = 132;
constexpr unsigned kWarps    = 1056;
constexpr unsigned kBlocks48 = 96;
constexpr unsigned kWarps48  = 768;

constexpr unsigned kStep = 256; // RowDot8's

TEST(RowSplit, GivesEveryRowOfQwenOneWarp) {
    for (const unsigned rows : kHeights) {
        EXPECT_TRUE(partitions(rows_of, rows, kWarps)) << rows << " rows";
        EXPECT_TRUE(partitions(rows_of, rows, kWarps48)) << rows << " rows";
    }
}

TEST(RowSplit, TheProofRejectsARunThatEndsEarly) {
    // Warp 5 stops a row short, so that row has no warp.
    const auto gap = [](unsigned warp, unsigned rows, unsigned warps) {
        const Rows r = rows_of(warp, rows, warps);
        return warp == 5 ? Rows{r.begin, r.end - 1} : r;
    };
    EXPECT_FALSE(partitions(gap, kVocab, kWarps));
    // The last warp stops a row short.
    const auto short_last = [](unsigned warp, unsigned rows, unsigned warps) {
        const Rows r = rows_of(warp, rows, warps);
        return warp + 1 == warps ? Rows{r.begin, r.end - 1} : r;
    };
    EXPECT_FALSE(partitions(short_last, kVocab, kWarps));
}

TEST(RowSplit, TheProofRejectsOverlappingRuns) {
    // Warp 5 also takes the last row of warp 4.
    const auto overlap = [](unsigned warp, unsigned rows, unsigned warps) {
        const Rows r = rows_of(warp, rows, warps);
        return warp == 5 ? Rows{r.begin - 1, r.end} : r;
    };
    EXPECT_FALSE(partitions(overlap, kVocab, kWarps));
    // Warp 5's run ends before it begins, and warp 6 starts at that end: warps 4 and 6 share
    // a row.
    const auto backwards = [](unsigned warp, unsigned rows, unsigned warps) {
        const Rows r = rows_of(warp, rows, warps);
        if (warp == 5) {
            return Rows{r.begin, r.begin - 1};
        }
        return warp == 6 ? Rows{rows_of(5, rows, warps).begin - 1, r.end} : r;
    };
    EXPECT_FALSE(partitions(backwards, kVocab, kWarps));
}

TEST(RowSplit, TheProofRejectsASplitThatSkipsTheFirstRow) {
    const auto late = [](unsigned warp, unsigned rows, unsigned warps) {
        const Rows r = rows_of(warp, rows, warps);
        return warp == 0 ? Rows{1, r.end} : r;
    };
    EXPECT_FALSE(partitions(late, kVocab, kWarps));
}

TEST(RowSplit, FitsIn32BitsExactlyWhereTheSplitHolds) {
    EXPECT_TRUE(mk::split_fits_32_bits(kVocab, kWarps));
    EXPECT_TRUE(mk::split_fits_32_bits(kVocab, kWarps48));
    // 28,268 * 151,936 is below 2^32, and 28,269 * 151,936 is above it.
    EXPECT_TRUE(mk::split_fits_32_bits(kVocab, 28268));
    EXPECT_FALSE(mk::split_fits_32_bits(kVocab, 28269));
    EXPECT_TRUE(partitions(rows_of, kVocab, 28268));
    EXPECT_FALSE(partitions(rows_of, kVocab, 28269));
}

TEST(RowKernel, ItsStepDividesEveryWidthOfQwen) {
    EXPECT_TRUE(mk::whole_steps(kEmbd, kStep));
    EXPECT_TRUE(mk::whole_steps(kQ, kStep));
    EXPECT_TRUE(mk::whole_steps(kFf, kStep));
}

TEST(RowKernel, AStepThatDoesNotDivideAWidthFails) {
    EXPECT_FALSE(mk::whole_steps(kEmbd, 768));
    EXPECT_FALSE(mk::whole_steps(kQ, 768));
    EXPECT_FALSE(mk::whole_steps(1000, kStep));
}

TEST(AttentionSplit, EachKvHeadHasABlock) {
    EXPECT_TRUE(mk::each_kv_head_has_a_block(kBlocks, kKvHeads));
    EXPECT_TRUE(mk::each_kv_head_has_a_block(kBlocks48, kKvHeads));
    EXPECT_TRUE(mk::each_kv_head_has_a_block(kKvHeads, kKvHeads));
    EXPECT_FALSE(mk::each_kv_head_has_a_block(kKvHeads - 1, kKvHeads));
}

TEST(AttentionMerge, NeedsOneThreadPerGroupDimension) {
    EXPECT_TRUE(mk::one_thread_per_group_dimension(kThreads, kGroup, kHeadDim));
    EXPECT_FALSE(mk::one_thread_per_group_dimension(128, kGroup, kHeadDim));
    EXPECT_FALSE(mk::one_thread_per_group_dimension(512, kGroup, kHeadDim));
    EXPECT_TRUE(mk::one_thread_per_group_dimension(512, 4, kHeadDim)); // Qwen3-4B: 32 / 8 heads
}

// The RTX 4070 Ti SUPER's shared memory, as cudaGetDeviceProperties reports it.
constexpr unsigned kSharedPerSm = 102400;
constexpr unsigned kReserved    = 1024; // per block

// The megakernel's shared memory for Qwen3-0.6B with blocks_per_sm blocks on each of 66 SMs.
// More blocks give a KV head more members, and the merge one more value per member.
constexpr std::uint64_t qwen_shared_bytes(unsigned blocks_per_sm) {
    return mk::megakernel_shared_bytes(kFf, kThreads / 32, kGroup, kHeadDim,
                                       66 * blocks_per_sm / kKvHeads);
}

TEST(SharedMemory, CountsTheBytesPtxasGivesQwen) {
    EXPECT_EQ(qwen_shared_bytes(2), 23008u); // the perf gate's smem
}

TEST(SharedMemory, UpToFourBlocksOfQwenFitAnSm) {
    for (unsigned blocks = 1; blocks <= 4; ++blocks) {
        EXPECT_TRUE(mk::fits_shared_memory(blocks, qwen_shared_bytes(blocks), kReserved,
                                           kSharedPerSm))
            << blocks << " blocks";
    }
    EXPECT_FALSE(mk::fits_shared_memory(5, qwen_shared_bytes(5), kReserved, kSharedPerSm));
}

TEST(SharedMemory, TheDriversReservePerBlockCounts) {
    // 4 blocks of 25,000 bytes are 100,000 bytes, and 104,096 with the reserve.
    EXPECT_TRUE(mk::fits_shared_memory(4, 25000, 0, kSharedPerSm));
    EXPECT_FALSE(mk::fits_shared_memory(4, 25000, kReserved, kSharedPerSm));
    // 4 * (24,576 + 1,024) is exactly 102,400.
    EXPECT_TRUE(mk::fits_shared_memory(4, 24576, kReserved, kSharedPerSm));
    EXPECT_FALSE(mk::fits_shared_memory(4, 24577, kReserved, kSharedPerSm));
}

// Qwen3-0.6B's weights, its BF16 KV cache, and the RTX 4070 Ti SUPER's device memory.
constexpr std::uint64_t kWeightBytes   = 1509347488 - 5951648; // file_size - data_start
constexpr std::uint64_t kKvBytesPerPos = 2 * 28 * kKvPerPos * 2; // K and V, 28 layers, BF16
constexpr std::uint64_t kDeviceMemory  = 16764764160;

TEST(DeviceMemory, QwenFitsUpTo133068Positions) {
    EXPECT_TRUE(mk::fits_device_memory(kWeightBytes, kKvBytesPerPos, 4096, kDeviceMemory));
    EXPECT_TRUE(mk::fits_device_memory(kWeightBytes, kKvBytesPerPos, 65536, kDeviceMemory));
    // 1,503,395,840 + 133,068 * 114,688 is below 16,764,764,160, one more position is above.
    EXPECT_TRUE(mk::fits_device_memory(kWeightBytes, kKvBytesPerPos, 133068, kDeviceMemory));
    EXPECT_FALSE(mk::fits_device_memory(kWeightBytes, kKvBytesPerPos, 133069, kDeviceMemory));
    EXPECT_FALSE(mk::fits_device_memory(kWeightBytes, kKvBytesPerPos, 1u << 20, kDeviceMemory));
}

TEST(DeviceMemory, WeightsLargerThanTheDeviceFailWithoutACache) {
    EXPECT_TRUE(mk::fits_device_memory(kDeviceMemory, kKvBytesPerPos, 0, kDeviceMemory));
    EXPECT_FALSE(mk::fits_device_memory(kDeviceMemory + 1, kKvBytesPerPos, 0, kDeviceMemory));
}

TEST(DeviceMemory, ACacheWhoseSizeWrapsIn64BitsFails) {
    // 2^62 positions of 4 bytes are 2^64 bytes, which a product would wrap to 0.
    EXPECT_FALSE(mk::fits_device_memory(0, 4, std::uint64_t{1} << 62, kDeviceMemory));
}

TEST(ContextLength, NCtxMayReachTheModelsContextLengthButNotPassIt) {
    EXPECT_TRUE(mk::within_context_length(4096, 32768));
    EXPECT_TRUE(mk::within_context_length(32768, 32768));
    EXPECT_FALSE(mk::within_context_length(32769, 32768));
    EXPECT_FALSE(mk::within_context_length(65536, 32768));
}

// Phases that declare only what they read and write.
template <unsigned Reads, unsigned Writes>
struct Phase {
    static constexpr unsigned reads  = Reads;
    static constexpr unsigned writes = Writes;
};
// The six phases of the megakernel's layer, by what they read and write.
using Qkv    = Phase<mk::kResidual, mk::kQkv>;
using Part   = Phase<mk::kQkv, mk::kPartials>;
using Merge  = Phase<mk::kPartials, mk::kKqv>;
using Output = Phase<mk::kKqv | mk::kResidual, mk::kFfnInp>;
using Ffn    = Phase<mk::kFfnInp, mk::kSwiglu>;
using Down   = Phase<mk::kSwiglu | mk::kFfnInp, mk::kLayerOut>;
using Layer  = PhaseList<Qkv, Part, Merge, Output, Ffn, Down>;

TEST(PhaseList, TheLayerHoldsEveryProof) {
    EXPECT_TRUE(mk::well_ordered(Layer{}));
    EXPECT_TRUE(mk::one_writer_per_buffer(Layer{}));
    EXPECT_TRUE(mk::writes_layer_output(Layer{}));
}

TEST(PhaseList, AMergeBeforeItsPartsIsNotWellOrdered) {
    using Swapped = PhaseList<Qkv, Merge, Part, Output, Ffn, Down>;
    EXPECT_FALSE(mk::well_ordered(Swapped{}));
    EXPECT_TRUE(mk::one_writer_per_buffer(Swapped{}));
    EXPECT_TRUE(mk::writes_layer_output(Swapped{}));
}

TEST(PhaseList, APhaseThatReadsABufferNobodyWroteIsNotWellOrdered) {
    // Without Ffn, nobody writes the SwiGLU that Down reads.
    using NoFfn = PhaseList<Qkv, Part, Merge, Output, Down>;
    EXPECT_FALSE(mk::well_ordered(NoFfn{}));
    EXPECT_TRUE(mk::one_writer_per_buffer(NoFfn{}));
}

TEST(PhaseList, TwoWritersOfOneBufferFail) {
    using Twice = PhaseList<Qkv, Qkv, Part, Merge, Output, Ffn, Down>;
    EXPECT_TRUE(mk::well_ordered(Twice{}));
    EXPECT_FALSE(mk::one_writer_per_buffer(Twice{}));
    // The residual stream has a writer already: the layer before.
    using Overwrite = PhaseList<Qkv, Part, Merge, Phase<mk::kKqv, mk::kResidual>>;
    EXPECT_FALSE(mk::one_writer_per_buffer(Overwrite{}));
}

TEST(PhaseList, ALayerThatNeverWritesItsOutputFails) {
    using NoDown = PhaseList<Qkv, Part, Merge, Output, Ffn>;
    EXPECT_TRUE(mk::well_ordered(NoDown{}));
    EXPECT_TRUE(mk::one_writer_per_buffer(NoDown{}));
    EXPECT_FALSE(mk::writes_layer_output(NoDown{}));
    EXPECT_FALSE(mk::writes_layer_output(PhaseList<>{}));
}

TEST(PhaseList, ADecodeStepOfQwenHas170Barriers) {
    EXPECT_EQ(mk::barriers(Layer{}, 28), 170u);
    EXPECT_EQ(mk::barriers(PhaseList<>{}, 28), 2u);
}
} // namespace
