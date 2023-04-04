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

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use crate::binom::binom_cdf;
use crate::data_interface::{ColumnIdentifiable, DataInterface, Response, Shadowable};
use crate::data_interface::y_bool::Y;
use crate::random_forest::RandomForest;
use crate::random_number_generator::factory::RngFactory;

const P_VALUE: f64 = 0.01;

pub fn boruta<T, U, SplitIndex>(df: T, y: U, pval_th: f64, max_runs: usize, ntree: usize) -> BorutaRes<SplitIndex>
where
    SplitIndex: ColumnIdentifiable + Clone + Copy + Send + Sync + Debug,
    SplitIndex::Col: Debug,
    T: Shadowable<SplitIndex, Y> + Sync + Send,
    U: Response<Y> + Sync + Send
{
    let mut iter = 0;

    let mut hits_map: HashMap<SplitIndex::Col, usize> = HashMap::new();
    for idx in df.get_col_ids() {
        hits_map.insert(idx,0);
    }

    let mut res = BorutaRes{
        tentative: df.get_col_ids(),
        confirmed: vec![],
        rejected: vec![]
    };

    while iter < max_runs && res.tentative.len() > 0 {
        iter += 1;
        println!("Iter {}", iter);
        let idxs = res.tentative.iter().cloned().chain(res.confirmed.iter().cloned()).collect();
        let mut cur_df = df.subset(&idxs);

        // Add shadow variables
        let rng_factory = RngFactory::new((iter+451256125) as u64, None, None); // TODO change static seed
        cur_df.add_shadows(rng_factory);

        // importance calculation
        let rf = RandomForest::new((iter+75754) as u64); // TODO should it be really static?
        let zscores = rf.zscore(&cur_df, &y, ntree, (cur_df.get_col_ids().len() as f64).sqrt().floor() as usize, false, None, None);

        let idxs_attr_set: HashSet<SplitIndex::Col> = HashSet::from_iter(idxs.iter().cloned());
        let idxs_all_set = HashSet::from_iter(cur_df.get_col_ids().iter().cloned());
        let idxs_shadow_set = &idxs_all_set-&idxs_attr_set;

        // when z-score > max shadow z-score add hit
        let max_shadow_zscore = idxs_shadow_set.iter()
            .map(|idx| zscores.get(idx).unwrap_or(&-1.))
            .max_by(|a, b| a.total_cmp(b)).unwrap();
        for idx in idxs.iter() {
            if zscores.get(idx).is_some() && zscores.get(idx).unwrap() > max_shadow_zscore {
                *hits_map.get_mut(idx).unwrap() += 1;
            }
        }

        // use binom to check if attr should be confirmed/rejected
        for idx in res.tentative.iter() {
            let hits = *hits_map.get(idx).unwrap();
            let pval_rej = binom_cdf(hits as u64, iter as u64, 0.5);
            if pval_rej < pval_th/(res.tentative.len() as f64) {
                // Add to rejected
                res.rejected.push(idx.clone());
            }

            if hits > 0 {
                let pval_conf = binom_cdf((hits-1) as u64, iter as u64, 0.5);
                if pval_conf > 1. - pval_th/(res.tentative.len() as f64) {
                    // Add to confirmed
                    res.confirmed.push(idx.clone());
                }
            }
        }

        // update tentative for further analysis
        let idxs_rejected: HashSet<SplitIndex::Col> = HashSet::from_iter(res.rejected.iter().cloned());
        let idxs_confirmed = HashSet::from_iter((res.confirmed.iter().cloned()));
        let idxs_tentative = &(&HashSet::from_iter((res.tentative.iter().cloned())) - &idxs_rejected) - &idxs_confirmed;
        res.tentative = idxs_tentative.into_iter().collect();
        println!("Tentative: {} Rejected: {} Confirmed: {}", res.tentative.len(), res.rejected.len(), res.confirmed.len());
    }

    return res;
}

#[derive(Debug)]
pub struct BorutaRes<SplitIndex: ColumnIdentifiable> {
    confirmed: Vec<SplitIndex::Col>,
    rejected: Vec<SplitIndex::Col>,
    tentative: Vec<SplitIndex::Col>
}

impl<SplitIndex: ColumnIdentifiable> BorutaRes<SplitIndex> {
    pub fn get_confirmed(&self) -> Vec<SplitIndex::Col> {
        self.confirmed.clone()
    }

    pub fn get_rejected(&self) -> Vec<SplitIndex::Col> {
        self.rejected.clone()
    }
}