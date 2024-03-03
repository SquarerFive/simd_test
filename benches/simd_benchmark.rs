struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }
}


// Basic matrix multiplication
fn multiply_basic(a: &Matrix, b: &Matrix, result: &mut Matrix) {
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// AVX2 matrix multiplication
#[cfg(target_arch = "x86_64")]
unsafe fn multiply_avx2(a: &Matrix, b: &Matrix, result: &mut Matrix) {
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = _mm256_setzero_ps();
            for k in (0..a.cols).step_by(8) {
                let a_vec = _mm256_loadu_ps(a.data[i * a.cols + k..].as_ptr());
                let b_vec = _mm256_loadu_ps(b.data[k * b.cols + j..].as_ptr());
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
            }
            let mut temp = [0.0f32; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), sum);
            result.set(i, j, temp.iter().sum());
        }
    }
}


use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use criterion::BenchmarkGroup;
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    // set measurement time
    let mut group: BenchmarkGroup<_> = c.benchmark_group("Matrix multiplication");
    // Set the target time for each benchmark to 5 seconds
    group.measurement_time(Duration::from_secs(60));


    let size = 256; // Example matrix size
    let a = Matrix::new(size, size);
    let b = Matrix::new(size, size);
    let mut result = Matrix::new(size, size);

    group.bench_function("Matrix multiplication (basic)", |bencher| {
        bencher.iter(|| multiply_basic(black_box(&a), black_box(&b), black_box(&mut result)))
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("Matrix multiplication (AVX2)", |bencher| {
        bencher.iter(|| unsafe { multiply_avx2(black_box(&a), black_box(&b), black_box(&mut result)) })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
