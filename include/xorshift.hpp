#pragma once
#include <cstdint>
#include <random>
#include <climits>
#include <cassert>
#include <cmath>

// Deterministic, stateless hashing utilities for reproducible pseudo-randomness.
// Useful for counter-based RNG: hash(seed, gen, individual, dim, tag, attempt, ...)
// and derive uniform values from the resulting 64-bit value.
namespace det_rng {
	// SplitMix64 finalizer / mixer. Public domain-quality mixing.
	inline uint64_t splitmix64(uint64_t x) {
		x += 0x9E3779B97F4A7C15ull;
		x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
		x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
		return x ^ (x >> 31);
	}

	inline uint64_t hash_init(uint64_t seed) {
		return splitmix64(seed);
	}

	inline uint64_t hash_append(uint64_t h, uint64_t v) {
		// Order-sensitive combine.
		return splitmix64(h ^ splitmix64(v));
	}

	inline uint64_t hash64(uint64_t a) {
		return splitmix64(a);
	}

	inline uint64_t hash64(uint64_t a, uint64_t b) {
		uint64_t h = hash_init(a);
		h = hash_append(h, b);
		return h;
	}

	inline uint64_t hash64(uint64_t a, uint64_t b, uint64_t c) {
		uint64_t h = hash_init(a);
		h = hash_append(h, b);
		h = hash_append(h, c);
		return h;
	}

	inline uint64_t hash64(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
		uint64_t h = hash_init(a);
		h = hash_append(h, b);
		h = hash_append(h, c);
		h = hash_append(h, d);
		return h;
	}

	inline uint64_t hash64(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e) {
		uint64_t h = hash_init(a);
		h = hash_append(h, b);
		h = hash_append(h, c);
		h = hash_append(h, d);
		h = hash_append(h, e);
		return h;
	}

	// Convert 64-bit value to double in [0,1) using top 53 bits.
	inline double u01_from_u64(uint64_t x) {
		// 53-bit precision: take the top 53 bits and scale.
		constexpr double inv_2pow53 = 1.0 / 9007199254740992.0; // 2^53
		return static_cast<double>(x >> 11) * inv_2pow53;
	}

	// Deterministic uniform integer in [0, bound) with minimal bias (multiply-high).
	inline uint32_t uniform_index(uint64_t x, uint32_t bound) {
		assert(bound > 0);
		__uint128_t prod = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(bound);
		return static_cast<uint32_t>(prod >> 64);
	}
}

static std::random_device seed_gen;
class xorshift {
	uint64_t seed;
public:
	xorshift() { seed = (uint64_t(seed_gen()) << 32) + seed_gen(); }
	inline uint64_t get64() {
		seed ^= (seed << 13); seed ^= (seed >> 7);
		return seed ^= (seed << 17);
	}
	xorshift(uint64_t arg) {
		// Expand input seed to a well-mixed internal state, allowing arg==0.
		seed = det_rng::splitmix64(arg);
		// xorshift has an absorbing all-zero state; guard the internal state, not the API seed.
		if (seed == 0) seed = 0x9E3779B97F4A7C15ull;
		for(int i=0; i<8; i++){
			seed ^= (seed << 13); seed ^= (seed >> 7); seed ^= (seed << 17);
		}
	}
	// [0, 2^64-1)
	inline uint32_t operator()() { return get64(); }
	// [0, r)
	inline uint32_t operator()(uint32_t r) { return operator()() % r; }
	// [mi, ma)
	inline int operator()(int mi, int ma) { return mi + operator()(ma - mi); }
	// [0,1)
	inline double prob() { return double(operator()()) / 0xffffffff; }

	inline double randbetExp(double lower, double upper) { 
	return lower * exp(log(upper / lower) * prob());
}
};
extern xorshift myRand;
inline double randbet(double lower, double upper) {
	return lower + (upper - lower) * myRand.prob();
}