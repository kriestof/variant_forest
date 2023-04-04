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


//TODO handle NA values
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::data_interface::{ColumnIdentifiable, DataInterface, Predicted, Response};
use crate::mask::Mask;
use crate::random_number_generator::factory::RngFactory;
use crate::random_number_generator::Rng;

type NodeHandle = usize;
type DfRowId = usize;
pub type ImportanceTree<T> = HashMap<T, i64>;

pub struct Tree<Y, SplitIndex> where
    SplitIndex: ColumnIdentifiable
{
    tree: Vec<Node<Y, SplitIndex>>,
    split_cols: HashSet<SplitIndex::Col>,
    mask_cache: Vec<Mask>,
    preds_cache: Vec<(Y, DfRowId)>,
    split_mask_map: HashMap<SplitIndex::Col, Vec<usize>>,
    split_idx_cache_range: Vec<(usize, usize)>,
    preds_cache_range: Vec<(usize, usize)>,
    rng: Rng,
    rng_factory: RngFactory,
    ith_tree: usize,
}

impl<Y, SplitIndex> Tree<Y, SplitIndex> where
    Y: Copy + Debug,
    SplitIndex: ColumnIdentifiable + Clone + Copy
{
    pub fn new(ith_tree: usize, rng_factory: &RngFactory) -> Tree<Y, SplitIndex> {
        let rng = rng_factory.new_rng_tree(ith_tree);
        Tree {
            tree: Vec::new(),
            split_cols: HashSet::new(),
            mask_cache: vec![],
            preds_cache: vec![],
            split_mask_map: HashMap::new(),
            split_idx_cache_range: vec![],
            preds_cache_range: vec![],
            rng,
            rng_factory: rng_factory.clone(),
            ith_tree
        }
    }

    #[inline]
    fn push_node(&mut self, node: Node<Y, SplitIndex>) -> NodeHandle {
        self.tree.push(node);
        return self.tree.len() - 1;
    }

    pub fn build_tree<T, U>(&mut self, df: &T, y: &U, mask: &Mask, mtry: usize, shadow_vars: bool, max_tree_depth: Option<usize>)
        where
            T: DataInterface<SplitIndex, Y>,
            U: Response<Y>
    {
        self._build_tree(df, y, &mask, mtry, shadow_vars, max_tree_depth, 0);
    }

    fn _build_tree<T, U>(&mut self, df: &T, y: &U, mask: &Mask, mtry: usize, shadow_vars: bool, max_tree_depth: Option<usize>, tree_depth: usize) -> NodeHandle
        where
            T: DataInterface<SplitIndex, Y>,
            U: Response<Y>
    {
        // If y is single class create node and return
        let class = y.get_class(&mask);

        if let Some(x) = class {
            let node = Node::create_leaf(x);
            return self.push_node(node);
        }

        if max_tree_depth.is_some() && tree_depth >= max_tree_depth.unwrap() {
            let major_class = y.get_major_class(&mask, &mut self.rng);
            let node = Node::create_leaf(major_class);
            return self.push_node(node);
        }

        // find best split
        let split_idx = df.find_min_idx(&mask, y, mtry, &mut self.rng, &self.rng_factory, shadow_vars);

        // Make split
        let masks = df.make_split(split_idx, &mask, &self.rng_factory, None);

        // If one split branch is empty terminate with leaf
        if masks[0].get_mask().len() == 0 || masks[1].get_mask().len() == 0 {
            let major_class = y.get_major_class(&mask, &mut self.rng);
            let node = Node::create_leaf(major_class);
            return self.push_node(node);
        }

        // Otherwise continue further down
        let l_node = self._build_tree(df, y, &masks[0], mtry, shadow_vars, max_tree_depth, tree_depth + 1);
        let r_node = self._build_tree(df, y, &masks[1], mtry, shadow_vars, max_tree_depth, tree_depth + 1);
        self.split_cols.insert(split_idx.get_col_id());
        let node = Node::create_split(split_idx, l_node, r_node);
        return self.push_node(node);
    }

    pub fn predict<T>(&mut self, df: &T, mask: &Mask, permuted_col: Option<SplitIndex::Col>, mask_ranks: &[usize]) -> Predicted<Y>
        where
            T: DataInterface<SplitIndex, Y>,
    {
        let mut preds = vec![None; mask.len()];

        match permuted_col {
            None => { let _ = self._predict_write_cache(df, mask, None, &mut preds, &mask_ranks, 0);},
            Some(_) => {
                let permuted_vec = df.permute_index(permuted_col.unwrap().clone(), &self.rng_factory, &mask, self.ith_tree);
                self._predict(df, mask, permuted_col, &permuted_vec, None, false, &mut preds, &mask_ranks)
            }
        };

        return preds.iter().map(|&x| x.unwrap()).collect();
    }

    fn _predict_write_cache<T>(&mut self, df: &T, mask: &Mask, node_id: Option<NodeHandle>, preds: &mut Vec<Option<Y>>, mask_ranks: &[usize], split_idx: usize) -> usize
        where
            T: DataInterface<SplitIndex, Y>
    {
        let node = match node_id {
            None => self.tree.last().unwrap().clone(),
            Some(nid) => self.tree[nid].clone()
        };

        match node {
            Node::Lf(leaf) => {
                self.mask_cache.push(mask.clone());
                self.preds_cache_range.push((self.preds_cache.len(), self.preds_cache.len()+mask.len()));

                for &i in mask.get_mask().iter() {
                    preds[mask_ranks[i]] = Some(leaf.get_class());

                    self.preds_cache.push((leaf.get_class(), mask_ranks[i]))
                }
                self.split_idx_cache_range.push((split_idx, split_idx));
                return split_idx;
            }

            Node::Sp(split) => {
                // make split
                let masks = df.make_split(split.split_index, &mask, &self.rng_factory, None);
                let preds_len_0 = self.preds_cache.len();

                let mut new_idx = self._predict_write_cache(df, &masks[0], Some(split.l_child_idx.clone()), preds, &mask_ranks, split_idx);
                new_idx = self._predict_write_cache(df, &masks[1], Some(split.r_child_idx.clone()), preds, &mask_ranks, new_idx);

                self.mask_cache.push(mask.clone());
                self.split_idx_cache_range.push((split_idx, new_idx));
                self.preds_cache_range.push((preds_len_0, self.preds_cache.len()));

                self.split_mask_map.entry(split.split_index.get_col_id())
                    .and_modify(|val| val.push(new_idx))
                    .or_insert(vec![new_idx]);
                return new_idx + 1;
            }
        }
    }

    #[inline]
    fn _preds_read_cache(&self, node_id: usize, permuted_col: &SplitIndex::Col, preds: &mut Vec<Option<Y>>) -> bool {
        let node = &self.tree[node_id];
        if let Node::Sp(_) = node {
            let should_get_from_cache = match self.split_mask_map.get(permuted_col) {
                None => true,
                Some(idxs) => {
                    for &idx in idxs.iter() {
                        if self.split_idx_cache_range[node_id].0 <= idx && self.split_idx_cache_range[node_id].1 >= idx {
                            return false;
                        }
                    }
                    true
                }
            };

            if should_get_from_cache {
                if self.preds_cache_range[node_id].0 < self.preds_cache.len() {
                    for i in self.preds_cache_range[node_id].0..self.preds_cache_range[node_id].1 {
                        let pred = self.preds_cache[i];
                        preds[pred.1] = Some(pred.0);
                    }
                    return true;
                }
            }
        }
        return false;
    }

    fn _predict<T>(&self, df: &T, mask: &Mask, permuted_col: Option<SplitIndex::Col>, permuted_vec: &T::InternalType, node_id: Option<NodeHandle>, altered: bool, preds: &mut Vec<Option<Y>>, mask_ranks: &[usize])
        where
            T: DataInterface<SplitIndex, Y>
    {
        let node = match node_id {
            None => self.tree.last().unwrap().clone(),
            Some(nid) => self.tree[nid].clone()
        };

        match node {
            Node::Lf(leaf) => {
                for &i in mask.get_mask().iter() {
                    preds[mask_ranks[i]] = Some(leaf.get_class());
                }
            }

            Node::Sp(split) => {
                // make split
                let permute = permuted_col.is_some() && permuted_col.unwrap() == split.split_index.get_col_id();
                let permuted_vec_arg = match permute {
                    true => Some(permuted_vec),
                    false => None
                };

                let masks_own;
                let masks = match self.mask_cache.len() > 0 && !permute && !altered {
                    true => [&self.mask_cache[split.l_child_idx], &self.mask_cache[split.r_child_idx]],
                    false => {
                        masks_own = df.make_split(split.split_index, &mask, &self.rng_factory, permuted_vec_arg);
                        [&masks_own[0], &masks_own[1]]
                    }
                };


                if permute || altered || !self._preds_read_cache(split.l_child_idx, &permuted_col.unwrap(), preds) {
                    self._predict(df, &masks[0], permuted_col, &permuted_vec, Some(split.l_child_idx.clone()), altered || permute, preds, &mask_ranks);
                }

                if permute || altered || !self._preds_read_cache(split.r_child_idx, &permuted_col.unwrap(), preds) {
                    self._predict(df, &masks[1], permuted_col, &permuted_vec, Some(split.r_child_idx.clone()), altered || permute, preds, &mask_ranks);
                }
            }
        }
    }

    pub fn importance<T, U>(&mut self, df: &T, y: &U, mask: &Mask) -> ImportanceTree<SplitIndex::Col>
        where
            T: DataInterface<SplitIndex, Y>,
            U: Response<Y>
    {
        let mut mask_ranks = vec![usize::MAX; y.len()];
        for (rank, &mask) in mask.get_mask().iter().enumerate() {
            mask_ranks[mask] = rank;
        }

        let preds = self.predict(df, &mask, None, &mask_ranks);
        let mut importance = ImportanceTree::new();
        let pred_err = y.pred_incorrect(&mask, &preds);

        for &col in self.split_cols.clone().iter() {
            let preds_perm = self.predict(df, &mask, Some(col.clone()), &mask_ranks);
            let pred_perm_err = y.pred_incorrect(&mask, &preds_perm);
            importance.insert(col, pred_perm_err as i64 - pred_err as i64);
        }
        return importance;
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
enum Node<T, U> {
    Sp(Split<U>),
    Lf(Leaf<T>),
}

#[derive(PartialEq, Eq, Debug, Clone)]
struct Split<T> {
    split_index: T,
    l_child_idx: NodeHandle,
    r_child_idx: NodeHandle,
}

#[derive(PartialEq, Eq, Debug, Clone)]
struct Leaf<T> {
    class: T,
}

impl<T: Copy> Leaf<T> {
    pub fn get_class(&self) -> T {
        return self.class;
    }
}

impl<T, U> Node<T, U> {
    fn create_split(split_index: U, l_child_idx: NodeHandle, r_child_idx: NodeHandle) -> Node<T, U> {
        Node::Sp(Split {
            split_index,
            l_child_idx,
            r_child_idx,
        })
    }

    fn create_leaf(class: T) -> Node<T, U> {
        Node::Lf(Leaf { class })
    }
}


#[cfg(test)]
mod tests {
    use crate::mask::Mask;
    use crate::data_interface::{ColumnIdentifiable, DataInterface, Permutable, Predicted, Response};
    use crate::random_number_generator::Rng;
    use crate::tree::{Node, Tree};
    use std::collections::{HashMap, HashSet};
    use std::marker::PhantomData;
    use crate::random_number_generator::factory::RngFactory;

    struct MyDf();

    #[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
    struct Sp(usize);

    #[derive(Copy, Clone)]
    struct Y();

    struct Void();

    impl ColumnIdentifiable for Sp {
        type Col = usize;

        fn get_col_id(&self) -> Self::Col {
            return self.0;
        }
    }

    impl DataInterface<Sp, usize> for MyDf {
        type InternalType = Void;

        fn get_ncol(&self) -> usize {
            unimplemented!();
        }

        fn find_min_idx<T>(&self, mask: &Mask, y: &T, mtry: usize, rng: &mut Rng, rng_factory: &RngFactory, shadow_vars: bool) -> Sp
            where T: Response<usize>
        {
            match mask.get_mask().as_slice() {
                &[1, 2, 3, 4, 5] => Sp(1),
                &[1, 2, 3] => Sp(2),
                _ => panic!("Unexpected mask in test Data Interface")
            }
        }

        fn make_split(&self, idx: Sp, mask: &Mask, rng_factory: &RngFactory, permute: Option<&Void>) -> [Mask; 2] {
            match mask.get_mask().as_slice() {
                &[1, 2, 3, 4, 5] => [Mask::new(vec![1, 2, 3]), Mask::new(vec![4, 5])],
                &[1, 2, 3] => [Mask::new(vec![1, 2]), Mask::new(vec![3])],
                &[1, 2, 5] => [Mask::new(vec![1, 2]), Mask::new(vec![5])],
                &[1, 2] => [Mask::new(vec![1, 2]), Mask::new(vec![])],
                &[5] => [Mask::new(vec![]), Mask::new(vec![5])],
                _ => panic!("Unexpected mask in test Data Interface")
            }
        }

        fn permute_index(&self, idx: usize, rng_factory: &RngFactory, oob_mask: &Mask, ith_tree: usize) -> Void {
            return Void();
        }
    }

    impl Response<usize> for Y {
        fn pred_incorrect(&self, mask: &Mask, preds: &Predicted<usize>) -> u64 {
            0
        }

        fn get_class(&self, mask: &Mask) -> Option<usize> {
            match mask.get_mask().as_slice() {
                &[1, 2] => Some(1),
                &[3] => Some(2),
                &[4, 5] => Some(3),
                _ => None
            }
        }

        fn get_major_class(&self, mask: &Mask, rng: &mut Rng) -> usize {
            unimplemented!();
        }

        fn pred_error(&self, mask: &Mask, preds: &Predicted<usize>) -> f64 {
            return 0.;
        }

        fn as_vector(&self) -> Vec<usize> {
            unimplemented!();
        }

        fn as_vector_ref(&self) -> &Vec<usize> {
            unimplemented!();
        }

        fn len(&self) -> usize { 6 }
    }

    #[test]
    fn build_tree() {
        let rng_factory = RngFactory::new(1, Some(100), Some(100));
        let mut tree = Tree::new(1, &rng_factory);
        let df = MyDf();
        let y = Y();
        let mask = Mask::new(vec![1, 2, 3, 4, 5]);

        tree.build_tree(&df, &y, &mask, 1, false, None);
        let expected_res = vec![
            Node::create_leaf(1 as usize),
            Node::create_leaf(2 as usize),
            Node::create_split(Sp(2usize), 0, 1),
            Node::create_leaf(3 as usize),
            Node::create_split(Sp(1usize), 2, 3),
        ];

        assert_eq!(tree.tree, expected_res)
    }

    #[test]
    fn predict() {
        let rng_factory = RngFactory::new(1, Some(100), Some(100));
        let mut tree = Tree::new(1, &rng_factory);
        tree.tree = vec![
            Node::create_leaf(1 as usize),
            Node::create_leaf(2 as usize),
            Node::create_split(Sp(2usize), 0, 1),
            Node::create_leaf(3 as usize),
            Node::create_split(Sp(1usize), 2, 3),
        ];

        let res = tree.predict(&MyDf(), &Mask::new(vec![1, 2, 5]), None, &vec![10, 0, 1, 10, 10, 2]);
        assert_eq!(res, vec![1, 1, 3])
    }

    #[test]
    fn importance() {
        let rng_factory = RngFactory::new(1, Some(100), Some(100));
        let mut tree = Tree::new(1, &rng_factory);
        tree.tree = vec![
            Node::create_leaf(1 as usize),
            Node::create_leaf(2 as usize),
            Node::create_split(Sp(2usize), 0, 1),
            Node::create_leaf(3 as usize),
            Node::create_split(Sp(1usize), 2, 3),
        ];
        tree.split_cols = HashSet::from([1usize, 2usize]);

        let res = tree.importance(&MyDf(), &Y(), &Mask::new(vec![1, 2, 3, 4, 5]));
        assert_eq!(res, HashMap::from([(1usize, 0), (2usize, 0)]));
    }
}