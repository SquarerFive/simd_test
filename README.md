# Rust AVX benchmark

## Getting started

1. Install Rust
2. Run `cargo bench --profile release`
3. Benchmark results will be in target/criterion

## Notes

Optimizations are explicitly disabled in Cargo.toml to show the performance difference between SIMD and SISD.
Without explicitly disabling it, the compiler will auto-vectorize operations.
