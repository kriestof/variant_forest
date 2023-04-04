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

const SAMPLE_FRACTION: f64 = 0.66;

use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::hash::Hash;
use std::thread;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::channel;

use crate::data_interface::{DataInterface, Response, ColumnIdentifiable};
use crate::mask::Mask;
use crate::random_number_generator::factory::RngFactory;
use crate::random_number_generator::Rng;
use crate::tree::{ImportanceTree, Tree};

pub type Importance<T> = HashMap<T, f64>;

pub struct RandomForest<Y, SplitIndex> {
    a: PhantomData<Y>,
    b: PhantomData<SplitIndex>,
    ntree: usize,
    seed: u64
}

impl <Y, SplitIndex> RandomForest<Y, SplitIndex> where
    Y:Copy + Send + Sync + Debug,
    SplitIndex: ColumnIdentifiable + Clone + Copy + Send + Sync
{
    pub fn new(seed: u64) -> Self {
        return RandomForest{
            a: PhantomData,
            b: PhantomData,
            ntree: 0,
            seed: seed
        };
    }

    fn next_tree<T, U>(&self, df: &T, y: &U , mtry: usize, shadow_vars: bool, rng_factory: &RngFactory, max_tree_depth: Option<usize>, ith_tree: usize) -> (Mask, Tree<Y, SplitIndex>,)
    where
        T: DataInterface<SplitIndex, Y> + Sync + Send,
        U: Response<Y> + Sync + Send
    {
        let mut tree = Tree::new(ith_tree, rng_factory);
        let mut rng = rng_factory.new_rng_tree_mask(ith_tree);
        let mask = Mask::random_mask(y.len(), SAMPLE_FRACTION, &mut rng);
        tree.build_tree(df, y, &mask, mtry, shadow_vars, max_tree_depth);
        return (mask, tree);
    }

    fn importance_per_tree<T, U> (&self, df: &T, y: &U, ntree: usize, mtry: usize, shadow_vars: bool, max_tree_depth: Option<usize>, multithread: Option<usize>) -> HashMap<SplitIndex::Col, Vec<i64>>
        where
            T: DataInterface<SplitIndex, Y> + Sync + Send,
            U: Response<Y> + Sync + Send
    {
        let rng_factory = RngFactory::new(
            self.seed,
            Some(df.get_ncol()),
            Some(ntree)
        );

        let mut imp: HashMap<SplitIndex::Col, Vec<i64>>;

        if multithread.is_some() {
            imp = HashMap::new();
            let thrs = multithread.unwrap(); // TODO this should be given by std::thread::available_parallelism
            let df_arc_tmp = Arc::new(df);
            let y_arc_tmp = Arc::new(y);
            let rng_factory_tmp= Arc::new(rng_factory);
            let (tx, rx) = channel();
            let ith_tree_mut = Arc::new(Mutex::new(0usize));

            thread::scope(|s| {
                for _ in 0..thrs {
                    let tx = tx.clone();
                    let ith_tree_mut = Arc::clone(&ith_tree_mut);
                    let y_arc = Arc::clone(&y_arc_tmp);
                    let df_arc = Arc::clone(&df_arc_tmp);
                    let rng_factory_arc = Arc::clone(&rng_factory_tmp);

                    s.spawn(move || {
                        loop {
                            let mut ith_tree_guard = ith_tree_mut.lock().unwrap();
                            let ith_tree = *ith_tree_guard;
                            if ith_tree >= ntree {
                                drop(ith_tree_guard);
                                break;
                            }
                            *ith_tree_guard += 1;
                            drop(ith_tree_guard); // unlock

                            let (mask, mut tree) = self.next_tree(*df_arc, *y_arc, mtry, shadow_vars, &*rng_factory_arc, max_tree_depth, ith_tree);
                            let oob_mask = mask.inverse(&(0..(*y_arc).len()).collect::<Vec<usize>>());

                            let tree_imp = tree.importance(*df_arc, *y_arc, &oob_mask);
                            tx.send(tree_imp).unwrap();
                        }
                    });
                }
            });
            for _ in 0..ntree {
                let tree_imp = rx.recv().unwrap();
                for (sp, val) in tree_imp.iter() {
                    imp.entry(*sp).and_modify(|row| {
                        row.push(*val);
                    }).or_insert(vec![*val]);
                }
            }
        } else {
            imp = HashMap::new();
            for ith_tree in 0..ntree {
                let (mask, mut tree) = self.next_tree(df, y, mtry, shadow_vars, &rng_factory, max_tree_depth, ith_tree);

                let oob_mask = mask.inverse(&(0..y.len()).collect::<Vec<usize>>());

                let tree_imp = tree.importance(df, y, &oob_mask);
                for (sp, val) in tree_imp.iter() {
                    imp.entry(*sp).and_modify(|row| {
                        row.push(*val)
                    }).or_insert(vec![*val]);
                }
            }
        }

        return imp;
    }

    pub fn zscore<T, U>(&self, df: &T, y: &U, ntree: usize, mtry: usize, shadow_vars: bool, max_tree_depth: Option<usize>, multithread: Option<usize>) -> Importance<SplitIndex::Col>
    where
        T: DataInterface<SplitIndex, Y> + Sync + Send,
        U: Response<Y> + Sync + Send
    {
        let imp_per_tree = self.importance_per_tree(df, y, ntree, mtry, shadow_vars, max_tree_depth, multithread);
        let mut res: Importance<SplitIndex::Col> = Importance::new();

        for (key, val) in imp_per_tree.iter() {
            let mean = val.iter().sum::<i64>() as f64 / val.len() as f64;
            let var = val.iter().map(|&x| (x as f64-mean).powi(2)).sum::<f64>() as f64/val.len() as f64;

            res.insert(key.clone(), mean/var.sqrt());
        }
        return res;
    }

    pub fn importance<T, U>(&self, df: &T, y: &U, ntree: usize, mtry: usize, shadow_vars: bool, max_tree_depth: Option<usize>, multithread: Option<usize>) -> Importance<SplitIndex::Col>
    where
        T: DataInterface<SplitIndex, Y> + Sync + Send,
        U: Response<Y> + Sync + Send
    {

        let imp_per_tree = self.importance_per_tree(df, y, ntree, mtry, shadow_vars, max_tree_depth, multithread);
        let mut res: Importance<SplitIndex::Col> = Importance::new();
        let oob_n = y.len() as f64 - (y.len() as f64 * SAMPLE_FRACTION).floor();

        for (key, val) in imp_per_tree.iter() {
            res.insert(key.clone(), val.iter().sum::<i64>() as f64 / val.len() as f64 / oob_n);
        }

        return res;
    }
}