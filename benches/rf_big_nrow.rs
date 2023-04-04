// Variant Forest
// Copyright (C) 2023 Krzysztof Piwoński <piwonski.kris@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::hash::Hasher;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, BenchmarkGroup};
use criterion::measurement::WallTime;

use variant_forest::data_interface::multi_x::{XDf, MultiX, ColSplitIndex, SplitColId};
use variant_forest::data_interface::y_bool::{YBool, Y};
use variant_forest::data_interface::three_val::ThreeValCol;
use variant_forest::random_forest::RandomForest;
use variant_forest::tree::Tree;
use variant_forest::mask::Mask;
use variant_forest::random_number_generator::Rng;

const SEED: u64 = 139547392210478;

fn new_threeval_col(x: &[i8]) -> MultiX {
    return MultiX::ThreeVal(ThreeValCol::new(x));
}

fn sample_0_1(rng: &mut Rng, k: usize) -> Vec<i8> {
    (0..k).map(|_| (rng.rand_uni() > 0.5) as i8).collect::<Vec<i8>>()
}

fn setup(nrow: usize, ncol: usize) -> (XDf, YBool) {
    let mut rng = Rng::new(SEED, 1);
    let xp1 = sample_0_1(&mut rng, nrow);
    let xp2 = sample_0_1(&mut rng, nrow);
    let xp3 = sample_0_1(&mut rng, nrow);

    let y_ins: Vec<bool> = xp1.iter().zip(xp2.iter()).map(|row| *row.0 == 1 && *row.1 == 1).collect();
    let y = YBool::new(&y_ins);


    let mut my_df_vec = vec![];

    for _ in 0..ncol {
        let xp = sample_0_1(&mut rng, nrow);
        let x = new_threeval_col(&xp);
        my_df_vec.push(x);
    }

    let my_df = XDf::new(my_df_vec);

    return (my_df, y);
}

fn rf_importance_performance_big_nrow(my_df: &XDf, y: &YBool, ntree: usize, multithred: Option<usize>) {
    let mut rf: RandomForest<Y, ColSplitIndex> = RandomForest::new(0);
    let res = rf.importance(my_df, y, ntree, 31, false, None, multithred, false);
}

fn bench_rayon(c: &mut Criterion) {
    let mut group = c.benchmark_group("rf big rayon");

    group.sample_size(10);

    const NTREE_BASE: usize = 50;
    let benches: Vec<(usize, usize)> = vec![
        (4000, 10_000),
        (8000, 10_000),
        (16_000, 50_000),
        (32_000, 50_000),
        (64_000, 100_000)
    ];

    for threads in [None, Some(1), Some(12), Some(24), Some(48), Some(96)] {
        for &bench in &benches {
            let (my_df, y) = setup(bench.0, bench.1);
            let thr = threads.unwrap_or(1);
            let thr_nam = threads.unwrap_or(0);
            group.bench_with_input(BenchmarkId::new(format!("multithread {thr_nam}"), bench.0), &NTREE_BASE, |b, &ntree| b.iter(|| rf_importance_performance_big_nrow(&my_df, &y, NTREE_BASE*thr, threads)));
        }
    }
    group.finish();
}

criterion_group!(benches, bench_rayon);
criterion_main!(benches);
