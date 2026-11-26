#include "gguf/reader.hpp"
#include "weights/verified_device.hpp"
#include "weights/weights.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <variant>

#include <gtest/gtest.h>

// Device checksums on the model the engine is built for (MODEL_GGUF). Needs a GPU.
namespace {
// Uploads data and returns what the device computes over it.
void device_checksums_of(mk::HostWeights data, mk::Checksums& out) {
    auto uploaded = mk::upload(data);
    ASSERT_TRUE(std::holds_alternative<mk::Upload>(uploaded))
        << std::get<mk::CudaError>(uploaded).call << ": " << std::get<mk::CudaError>(uploaded).text;
    const auto result = mk::device_checksums(std::get<mk::Upload>(uploaded).weights.view());
    ASSERT_TRUE(std::holds_alternative<mk::Checksums>(result))
        << std::get<mk::CudaError>(result).call << ": " << std::get<mk::CudaError>(result).text;
    out = std::get<mk::Checksums>(result);
}

struct WeightsTest : testing::Test {
    gguf::MappedFile           file;
    std::span<const std::byte> bytes;

    void SetUp() override {
        auto opened = gguf::MappedFile::open(MODEL_GGUF);
        ASSERT_TRUE(opened.has_value()) << MODEL_GGUF;
        file  = std::move(*opened);
        bytes = file.bytes();
        ASSERT_EQ(bytes.size(), model::file_size);
    }
};
} // namespace

TEST_F(WeightsTest, DeviceChecksumsMatchTheFile) {
    const mk::HostWeights weights = bytes.subspan(model::data_start).first<mk::weights_size>();
    mk::Checksums         device{};
    ASSERT_NO_FATAL_FAILURE(device_checksums_of(weights, device));
    EXPECT_EQ(device.tensors, mk::host_checksums(weights));
    if (mk::layer_count > 0) {
        EXPECT_EQ(device.layer_q, mk::checksum(weights.subspan(mk::layer_probe.offset,
                                                               mk::layer_probe.bytes)));
    }
}

// The negative control. The upload starts 32 bytes early, so every tensor's device address
// holds the file's bytes from 32 bytes before the tensor: the device reads at a wrong offset.
TEST_F(WeightsTest, AnOffsetWrongBy32BytesDoesNotMatch) {
    constexpr std::size_t kShift = 32;
    ASSERT_GE(model::data_start, kShift);
    const mk::HostWeights right = bytes.subspan(model::data_start).first<mk::weights_size>();
    const mk::HostWeights shifted =
        bytes.subspan(model::data_start - kShift).first<mk::weights_size>();
    mk::Checksums device{};
    ASSERT_NO_FATAL_FAILURE(device_checksums_of(shifted, device));

    const auto  expected = mk::host_checksums(right);
    std::size_t matched  = 0;
    for (std::size_t i = 0; i < mk::tensor_count; ++i) {
        matched += device.tensors[i] == expected[i];
    }
    EXPECT_EQ(matched, 0u) << "of " << mk::tensor_count << " tensors";
    // The device did read what was uploaded: the sums are those of the shifted bytes.
    EXPECT_EQ(device.tensors, mk::host_checksums(shifted));
}

// No device has 0 SMs, so a build for this GPU is refused on every machine.
struct NoSmGpu {
    static constexpr const char*   name                      = "no-SM GPU";
    static constexpr unsigned      sms                       = 0;
    static constexpr unsigned      cuda_arch                 = 89;
    static constexpr std::uint64_t device_memory             = 0;
    static constexpr unsigned      shared_per_sm             = 0;
    static constexpr unsigned      shared_reserved_per_block = 0;
};

TEST(VerifiedDevice, RefusesADeviceWithAnotherSmCount) {
    const auto device = mk::VerifiedDevice<NoSmGpu>::check();
    ASSERT_FALSE(device.has_value());
    EXPECT_TRUE(device.error().starts_with("built for the no-SM GPU (0 SMs), this GPU has "))
        << device.error();
}
