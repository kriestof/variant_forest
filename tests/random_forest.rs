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

use serde_json::json;
use float_cmp::assert_approx_eq;

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

#[test]
fn rf_importance_mtry_1() {
    let mut rng = Rng::new(SEED, 1);
    let xp1 = sample_0_1(&mut rng, 100);
    let xp2 = sample_0_1(&mut rng, 100);
    let xp3 = sample_0_1(&mut rng, 100);
    let y = YBool::new(&xp1
        .iter().map(|&x| x == 1).collect::<Vec<bool>>());
    let x1 = new_threeval_col(&xp1);
    let x2 = new_threeval_col(&xp2);
    let x3 = new_threeval_col(&xp3);
    let my_df = XDf::new(vec![x1, x2, x3]);

    let mut rf: RandomForest<Y, ColSplitIndex> = RandomForest::new(0);
    let res = rf.importance(&my_df, &y, 1000, 1, false, None, None);
    assert!(*res.get(&SplitColId{col_id: 0, shadow: false}).unwrap() > 0.30);
    assert_approx_eq!(f64, *res.get(&SplitColId{col_id: 1, shadow: false}).unwrap(), 0., epsilon=0.02);
    assert_approx_eq!(f64, *res.get(&SplitColId{col_id: 2, shadow: false}).unwrap(), 0., epsilon=0.02);
}

// #[test]
// fn rf_importance_mtry_1_shadow() {
//     let mut rng = Rng::new(SEED, 1);
//     let xp1 = sample_0_1(&mut rng, 100);
//     let xp2 = sample_0_1(&mut rng, 100);
//     let xp3 = sample_0_1(&mut rng, 100);
//     let y = YBool::new(&xp1
//         .iter().map(|&x| x == 1).collect::<Vec<bool>>());
//     let x1 = new_threeval_col(&xp1);
//     let x2 = new_threeval_col(&xp2);
//     let x3 = new_threeval_col(&xp3);
//     let my_df = XDf::new(vec![x1, x2, x3]);
//
//     let mut rf: RandomForest<Y, ColSplitIndex> = RandomForest::new(0);
//     let res = rf.importance(&my_df, &y, 1000, 1, true);
//
//     assert!(*res.get(&SplitColId{col_id: 0, shadow: false}).unwrap() > 0.2);
//     assert_approx_eq!(f64, *res.get(&SplitColId{col_id: 0, shadow: true}).unwrap(), 0., epsilon=0.02);
// }

#[test]
fn rf_importance_interactions() {
    let mut rng = Rng::new(SEED, 1);
    let xp1 = sample_0_1(&mut rng, 100);
    let xp2 = sample_0_1(&mut rng, 100);
    let xp3 = sample_0_1(&mut rng, 100);

    let y_ins: Vec<bool> = xp1.iter().zip(xp2.iter()).map(|row| *row.0 == 1 && *row.1 == 1).collect();
    let y = YBool::new(&y_ins);


    let x1 = new_threeval_col(&xp1);
    let x2 = new_threeval_col(&xp2);
    let x3 = new_threeval_col(&xp3);

    let mut my_df_vec = vec![x1, x2];

    for i in 1..100 {
        let xp = sample_0_1(&mut rng, 100);
        let x = new_threeval_col(&xp);
        my_df_vec.push(x);
    }
    let my_df = XDf::new(my_df_vec);

    let mut rf: RandomForest<Y, ColSplitIndex> = RandomForest::new(0);
    let res = rf.importance(&my_df, &y, 1000, 10, false, None, None);
    assert!(*res.get(&SplitColId{col_id: 0, shadow: false}).unwrap() > 0.04);
    assert!(*res.get(&SplitColId{col_id: 1, shadow: false}).unwrap() > 0.04);
    for i in 2..100 {
        assert_approx_eq!(f64, *res.get(&SplitColId{col_id: i, shadow: false}).unwrap(), 0., epsilon=0.02);
    }
}

fn serde_array_to_three_val(x: &serde_json::Value) -> MultiX {
    let arr_i8 = x.as_array().unwrap().iter()
        .map(|x| x.as_i64().unwrap() as i8)
        .collect::<Vec<i8>>();
    new_threeval_col(&arr_i8)
}

#[test]
fn rf_importance_srx() {
    let data_str = "{\"A\":[2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],\"B\":[2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1],\"N1\":[2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1],\"N2\":[2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1],\"N3\":[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"AoB\":[2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1],\"AnB\":[2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1],\"nA\":[1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],\"Y\":[false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false]}";
    let df_json: serde_json::Value = serde_json::from_str(data_str).unwrap();
    let y_vec: Vec<bool> = df_json["Y"].as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect();
    let y = YBool::new(&y_vec);


    let A = serde_array_to_three_val(&df_json["A"]);
    let B = serde_array_to_three_val(&df_json["B"]);
    let N1 = serde_array_to_three_val(&df_json["N1"]);
    let N2 = serde_array_to_three_val(&df_json["N2"]);
    let N3 = serde_array_to_three_val(&df_json["N3"]);
    let AoB = serde_array_to_three_val(&df_json["AoB"]);
    let AnB = serde_array_to_three_val(&df_json["AnB"]);
    let nA = serde_array_to_three_val(&df_json["nA"]);

    let my_df = XDf::new(vec![A, B, N1, N2, N3, AoB, AnB, nA]);

    let mut rf: RandomForest<Y, ColSplitIndex> = RandomForest::new(0);
    let res = rf.importance(&my_df, &y, 1000, 3, false, None, None);
    assert_approx_eq!(f64, *res.get(&SplitColId{col_id: 2, shadow: false}).unwrap(), 0., epsilon=0.06);
    assert_approx_eq!(f64, *res.get(&SplitColId{col_id: 3, shadow: false}).unwrap(), 0., epsilon=0.06);
    assert_approx_eq!(f64, *res.get(&SplitColId{col_id: 4, shadow: false}).unwrap(), 0., epsilon=0.06);

    assert!(*res.get(&SplitColId{col_id: 0, shadow: false}).unwrap() > 0.1);
    assert!(*res.get(&SplitColId{col_id: 1, shadow: false}).unwrap() > 0.1);
    assert!(*res.get(&SplitColId{col_id: 5, shadow: false}).unwrap() > 0.1);
    assert!(*res.get(&SplitColId{col_id: 6, shadow: false}).unwrap() > 0.1);
    assert!(*res.get(&SplitColId{col_id: 7, shadow: false}).unwrap() > 0.1);
}


#[test]
fn it_does_not_predict_xor_with_max_tree_depth_1() {
    let mut rng = Rng::new(SEED, 1);
    let xp1 = sample_0_1(&mut rng, 100);
    let xp2 = sample_0_1(&mut rng, 100);
    let xp3 = sample_0_1(&mut rng, 100);
    let y_vec: Vec<Y> = xp1.iter().zip(xp2.iter())
        .map(|row| ((*row.0 == 0) ^ (*row.1 == 0)) as bool).collect();

    let y = YBool::new(&y_vec);
    let x1 = new_threeval_col(&xp1);
    let x2 = new_threeval_col(&xp2);
    let x3 = new_threeval_col(&xp3);
    let my_df = XDf::new(vec![x1, x2, x3]);

    let mut rf: RandomForest<Y, ColSplitIndex> = RandomForest::new(0);

    let res = rf.importance(&my_df, &y, 1000, 1, false, Some(1), None);
    assert!(*res.get(&SplitColId{col_id: 0, shadow: false}).unwrap() < 0.05);
    let res = rf.importance(&my_df, &y, 1000, 1, false, Some(2), None);
    assert!(*res.get(&SplitColId{col_id: 0, shadow: false}).unwrap() > 0.1);
}


// #[test]
// fn rf_importance_srx_shadow() {
//     let data_str = "{\"A\":[2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],\"B\":[2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1],\"N1\":[2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1],\"N2\":[2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1],\"N3\":[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"AoB\":[2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1],\"AnB\":[2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1],\"nA\":[1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],\"Y\":[false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false,false,true,true,false]}";
//     let df_json: serde_json::Value = serde_json::from_str(data_str).unwrap();
//     let y_vec: Vec<bool> = df_json["Y"].as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect();
//     let y = YBool::new(&y_vec);
//
//
//     let A = serde_array_to_three_val(&df_json["A"]);
//     let B = serde_array_to_three_val(&df_json["B"]);
//     let N1 = serde_array_to_three_val(&df_json["N1"]);
//     let N2 = serde_array_to_three_val(&df_json["N2"]);
//     let N3 = serde_array_to_three_val(&df_json["N3"]);
//     let AoB = serde_array_to_three_val(&df_json["AoB"]);
//     let AnB = serde_array_to_three_val(&df_json["AnB"]);
//     let nA = serde_array_to_three_val(&df_json["nA"]);
//
//     let my_df = XDf::new(vec![A, B, N1, N2, N3, AoB, AnB, nA]);
//
//     let mut rf: RandomForest<Y, ColSplitIndex> = RandomForest::new(0);
//     let res = rf.importance(&my_df, &y, 1000, 4, true);
//
//     assert!(*res.get(&SplitColId{col_id: 5, shadow: false}).unwrap() > 0.15);
//     assert_approx_eq!(f64, *res.get(&SplitColId{col_id: 5, shadow: true}).unwrap(), 0., epsilon=0.03);
// }

// #[test]
// fn tree_importance() {
//     let mut rng = Rng::new(1);
//     let xp1 = sample_0_1(&mut rng, 100);
//     let y = YBool::new(&xp1
//         .iter().map(|&x| x == 1).collect::<Vec<bool>>());
//     let x1 = new_threeval_col(&xp1);
//     let x2 = new_threeval_col(&sample_0_1(&mut rng, 100));
//     let x3 = new_threeval_col(&sample_0_1(&mut rng, 100));
//     let my_df = XDf::new(vec![x1, x2, x3]);
//
//
//     let mut tree: Tree<Y, ColSplitIndex> = Tree::new(3);
//     let res = tree.build_tree(&my_df, &y, &Mask::new([6,5,4,3,2,1,0].to_vec()), 3, false);
//     let res = tree.importance(&my_df, &y, &Mask::new([9,8,7].to_vec()));
//     dbg!(res);
//     panic!()
// }