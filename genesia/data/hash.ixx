export module genesia.hash;
import std;
export namespace genesia {
    struct Sha256 final {
        std::array<std::uint32_t, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::array<unsigned char, 64> buffer{};
        std::uint64_t bytes{};
        void update(std::span<const unsigned char> input);
        std::array<unsigned char, 32> finish();

    private:
        void compress(const unsigned char* input);
    };
    std::string sha256(std::span<const unsigned char> bytes);
} // namespace genesia
