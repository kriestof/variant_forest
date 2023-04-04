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

// use serde_json::json;
use float_cmp::assert_approx_eq;

use variant_forest::data_interface::multi_x::{XDf, MultiX, ColSplitIndex, SplitColId};
use variant_forest::data_interface::y_bool::{YBool, Y};
use variant_forest::data_interface::three_val::ThreeValCol;
use variant_forest::random_forest::RandomForest;
use variant_forest::boruta::{boruta, BorutaRes};
// use variant_forest::tree::Tree;
// use variant_forest::mask::Mask;
use variant_forest::random_number_generator::Rng;

const SEED: u64 = 139547392210478;

fn new_threeval_col(x: &[i8]) -> MultiX {
    return MultiX::ThreeVal(ThreeValCol::new(x));
}

fn sample_0_1(rng: &mut Rng, k: usize) -> Vec<i8> {
    (0..k).map(|_| (rng.rand_uni() > 0.5) as i8).collect::<Vec<i8>>()
}

#[test]
fn boruta_interactions() {
    let mut rng = Rng::new(SEED, 1);
    let xp1 = sample_0_1(&mut rng, 1000);
    let xp2 = sample_0_1(&mut rng, 1000);
    // let xp3 = sample_0_1(&mut rng, 100);

    let y_ins: Vec<bool> = xp1.iter().zip(xp2.iter()).map(|row| *row.0 == 1 && *row.1 == 1).collect();
    let y = YBool::new(&y_ins);


    let x1 = new_threeval_col(&xp1);
    let x2 = new_threeval_col(&xp2);

    let mut my_df_vec = vec![x1, x2];

    for i in 1..100 {
        let xp = sample_0_1(&mut rng, 1000);
        let x = new_threeval_col(&xp);
        my_df_vec.push(x);
    }
    let my_df = XDf::new(my_df_vec);

    let boruta_res: BorutaRes<ColSplitIndex> = boruta(my_df, y, 0.01, 100, 500);

    let mut res_confirmed = boruta_res.get_confirmed().iter()
        .map(|split_col| split_col.col_id)
        .collect::<Vec<_>>();
    res_confirmed.sort();
    assert_eq!(res_confirmed, [0, 1]);


    let mut res_rejected = boruta_res.get_rejected().iter()
        .map(|split_col| split_col.col_id)
        .collect::<Vec<_>>();
    res_rejected.sort();
    assert!(res_rejected.len() > 95)
}

fn serde_array_to_three_val(x: &serde_json::Value) -> MultiX {
    let arr_i8 = x.as_array().unwrap().iter()
        .map(|x| x.as_i64().unwrap() as i8)
        .collect::<Vec<i8>>();
    new_threeval_col(&arr_i8)
}

#[test]
fn boruta_srx() {
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

    let boruta_res: BorutaRes<ColSplitIndex> = boruta(my_df, y, 0.05, 100, 1000);
    let mut res_confirmed = boruta_res.get_confirmed().iter()
        .map(|split_col| split_col.col_id)
        .collect::<Vec<_>>();
    res_confirmed.sort();
    assert_eq!(res_confirmed, [0, 1, 5, 6, 7]);

    let mut res_rejected = boruta_res.get_rejected().iter()
        .map(|split_col| split_col.col_id)
        .collect::<Vec<_>>();
    res_rejected.sort();
    assert_eq!(res_rejected, [2, 3, 4]);
}