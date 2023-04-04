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

use crate::mask::Mask;
use crate::data_interface::three_val::{ThreeValCol, ThreeValPivot};
use crate::data_interface::{DataInterface, Response, ColumnIdentifiable, Splittable, Permutable, Shadowable};
use crate::random_number_generator::Rng;
use crate::data_interface::y_bool::Y;
use crate::random_number_generator::factory::RngFactory;

#[derive(Debug, PartialEq, Eq)]
pub struct XDf {
    data: Vec<MultiX>,
    idx_to_splitid_map: Vec<usize>,
    splitid_to_idx_map: Vec<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiX {
    ThreeVal(ThreeValCol)
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum MultiPivot {
    ThreeVal(ThreeValPivot)
}

#[derive(Copy, Clone, Debug)]
pub struct ColSplitIndex {
    pub col_id: usize,
    pub pivot: MultiPivot,
    pub shadow: bool //TODO remove
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub struct SplitColId {
    pub col_id: usize,
    pub shadow: bool //TODO remove
}

impl ColumnIdentifiable for ColSplitIndex {
    type Col = SplitColId;

    #[inline]
    fn get_col_id(&self) -> Self::Col {
        return SplitColId{col_id: self.col_id, shadow: self.shadow};
    }
}

impl XDf {
    pub fn new(cols: Vec<MultiX>) -> XDf {
        let idx_to_splitid_map: Vec<usize> = (0..cols.len()).collect();
        let splitid_to_idx_map: Vec<usize> = (0..cols.len()).collect();
        return XDf{data: cols, idx_to_splitid_map, splitid_to_idx_map};
    }

    #[inline]
    fn get_col(&self, col: usize) -> &MultiX {
        return &self.data[col];
    }

    fn idx_to_splitid(&self, idx: usize) -> usize {
        self.idx_to_splitid_map[idx]
    }

    fn splitid_to_idx(&self, splitid: usize) -> usize {
        self.splitid_to_idx_map[splitid]
    }
}

impl Shadowable<ColSplitIndex, Y> for XDf {
    fn subset(&self, split_ids: &Vec<SplitColId>) -> Self {
        let idx_to_splitid_map: Vec<usize> = split_ids.iter().map(|x| x.col_id).collect();
        let mut splitid_to_idx_map = self.splitid_to_idx_map.clone();
        for (idx, &col_id) in idx_to_splitid_map.iter().enumerate() {
            splitid_to_idx_map[col_id] = idx;
        }

        XDf {
            data: split_ids.iter().map(|&col_id| self.data[self.splitid_to_idx(col_id.col_id)].clone()).collect(),
            idx_to_splitid_map,
            splitid_to_idx_map,
        }
    }

    fn add_shadows(&mut self, rng_factory: RngFactory) {
        let mask = Mask::new((0..self.data[0].len()).collect());
        let max_splitid = *self.idx_to_splitid_map.iter().max().unwrap();


        let mut num_shadow = self.data.len();
        while num_shadow < 5 {
            num_shadow *= 2;
        }
        self.splitid_to_idx_map.resize(max_splitid+num_shadow+1, 0);

        for i in 0..num_shadow {
            let mut rng = rng_factory.new_rng_shadow(i);
            self.data.push(self.data[i % self.data.len()].permute(rng, &mask));
            self.idx_to_splitid_map.push(max_splitid+i+1);
            self.splitid_to_idx_map[max_splitid+i+1] = self.idx_to_splitid_map.len()-1;
        }
    }

    fn get_col_ids(&self) -> Vec<SplitColId> {
        self.idx_to_splitid_map.iter().map(|&i| SplitColId{col_id: i, shadow: false}).collect()
    }
}

impl DataInterface<ColSplitIndex, Y> for XDf {
    type InternalType = MultiX;
    #[inline]
    fn get_ncol(&self) -> usize {
        return self.data.len();
    }

    fn find_min_idx<U>(&self, mask: &Mask, y: &U, mtry: usize, rng: &mut Rng, rng_factory: &RngFactory, shadow_vars: bool) -> ColSplitIndex
    where
        U: Response<Y>
    {
        let range: Vec<usize>;
        if shadow_vars {
            range = (0..self.get_ncol()*2).collect();
        } else {
            range = (0..self.get_ncol()).collect();
        }

        let min_idx = rng.sample(&range, mtry).iter_mut().map(|col| {
            let mut shadow_rng = None;
            if *col >= self.get_ncol() {
                *col = *col-self.get_ncol();
                shadow_rng = Some(rng_factory.new_rng_shadow(*col));
            }


            let x = self.get_col(*col);
            let res = x.gen_optimal_pivot(&mask, y, shadow_rng);
            return (res.0, res.1, self.idx_to_splitid(*col), shadow_rng.is_some());
        }).min_by(|x, y| x.1.partial_cmp(&y.1).expect("Gini score has strange value (NaN like)"));

        let min_idx_un = min_idx.unwrap();
        return ColSplitIndex {col_id: min_idx_un.2, pivot: min_idx_un.0, shadow: min_idx_un.3};
    }

    fn make_split(&self, idx: ColSplitIndex, mask: &Mask, rng_factory: &RngFactory, permuted_vec: Option<&MultiX>) -> [Mask; 2] {
        let col = match permuted_vec {
            Some(x) => x,
            None => self.get_col(self.splitid_to_idx(idx.col_id))
        };

        let shadow_rng = match idx.shadow {
            true => Some(rng_factory.new_rng_shadow(idx.col_id)),
            false => None
        };

        return col.split_with_pivot(&mask, &idx.pivot, shadow_rng);
    }

    fn permute_index(&self, col_id: SplitColId, rng_factory: &RngFactory, oob_mask: &Mask, ith_tree: usize) -> MultiX {
        let col = self.get_col(self.splitid_to_idx(col_id.col_id));
        let rng = rng_factory.new_rng_permutation(ith_tree, col_id.col_id);
        return col.permute(rng, oob_mask);
    }
}

impl Permutable for MultiX {
    fn permute(&self, mut perm_rng: Rng, oob_mask: &Mask) -> Self {
        match self {
            MultiX::ThreeVal(x) => MultiX::ThreeVal(x.permute(perm_rng, oob_mask)),
            _ =>  panic!("Incoherent X")
        }
    }
}

impl Splittable<Y> for MultiX {
    type Pivot = MultiPivot;

    fn split_with_pivot(&self, mask: &Mask, p: &Self::Pivot, shadow_rng: Option<Rng>) -> [Mask; 2] {
        match (self, p) {
            (MultiX::ThreeVal(x), MultiPivot::ThreeVal(p)) => x.split_with_pivot(&mask, &p, shadow_rng),
            _ =>  panic!("Incoherent X -- pivot mixture")
        }
    }

    fn gen_optimal_pivot<T>(&self, mask: &Mask, y:  &T, perm_seed_shadow: Option<Rng>) -> (Self::Pivot, f64)
    where
        T: Response<Y>
    {
        match self {
            MultiX::ThreeVal(x) => {
                let (piv, score) = x.gen_optimal_pivot(&mask, y, perm_seed_shadow);
                (MultiPivot::ThreeVal(piv), score)
            }
        }
    }
}

impl MultiX {
    pub fn len(&self) -> usize{
        match self {
            MultiX::ThreeVal(x) => x.len()
        }
    }
}


#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use crate::mask::Mask;
    use crate::data_interface::three_val::{ThreeValCol, ThreeValPivot};
    use crate::data_interface::{DataInterface, Shadowable, Splittable};
    use crate::data_interface::multi_x::{MultiPivot, MultiX, ColSplitIndex, XDf, SplitColId};
    use crate::random_number_generator::Rng;
    use crate::data_interface::y_bool::YBool;
    use crate::random_number_generator::factory::RngFactory;

    #[test]
    fn split_with_pivot_multi_x() {
        let x_vec = ThreeValCol::new(&vec![0, 0, 1, 2, 2, 1, 0, 1]);
        let mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6]);
        let mult_x = MultiX::ThreeVal(x_vec);
        assert_eq!(mult_x.split_with_pivot(&mask, &MultiPivot::ThreeVal(ThreeValPivot::NotRed), None),
                   [Mask::new(vec![2, 3, 4, 5]), Mask::new(vec![0, 1, 6])]);
    }

    #[test]
    fn gen_optimal_pivot_multi_x() {
        let x = MultiX::ThreeVal(ThreeValCol::new(&vec![0, 2, 2, 1, 1, 0, 2, 0, 1]));
        let y = YBool::new(&vec![false, true, true, false, true, false, true, true, false]);
        let (piv, score) = x.gen_optimal_pivot(&Mask::new((0..=8).collect()), &y, None);
        assert_eq!(piv, MultiPivot::ThreeVal(ThreeValPivot::NotBlue));
        assert_approx_eq!(f64, score, 6./9. - (4*4+2*2) as f64/6./9.)
    }

    #[test]
    fn find_min_idx_df() {
        let x1 = MultiX::ThreeVal(ThreeValCol::new(&vec![0, 2, 2, 1, 1, 0, 2, 0, 1]));
        let x2 = MultiX::ThreeVal(ThreeValCol::new(&vec![0, 1, 2, 0, 1, 0, 1, 2, 0]));
        let df = XDf::new(vec![x1, x2]);
        let y = YBool::new(&vec![false, true, true, false, true, false, true, true, false]);
        let mask = &Mask::new((0..=8).collect());
        let res = df.find_min_idx(&mask,
                                  &y,
                                  2,
                                  &mut Rng::new(4, 1),
                                  &RngFactory::new(1,
                                  Some(100),
                                  Some(100)),
                                  false);
        assert_eq!(res.col_id, 1);
        assert_eq!(res.pivot, MultiPivot::ThreeVal(ThreeValPivot::NotRed));
    }

    // #[test]
    // fn find_min_idx_df_shadow_vars() {
    //     let x1 = MultiX::ThreeVal(ThreeValCol::new(&vec![0, 2, 2, 1, 1, 0, 2, 0, 1]));
    //     let x2 = MultiX::ThreeVal(ThreeValCol::new(&vec![0, 1, 2, 0, 1, 0, 1, 2, 0]));
    //     let df = XDf::new(vec![x1, x2]);
    //     let y = YBool::new(&vec![false, false, false, false, true, false, true, true, true]);
    //     let mask = &Mask::new((0..=8).collect());
    //     let res = df.find_min_idx(&mask,
    //                               &y,
    //                               4,
    //                               &mut Rng::new(2, 1),
    //                               &RngFactory::new(1,
    //                                                Some(100),
    //                                                Some(100)),
    //                               true);
    //     assert_eq!(res.col_id, 1);
    //     assert_eq!(res.pivot, MultiPivot::ThreeVal(ThreeValPivot::NotGreen));
    // }

    #[test]
    fn make_split_df() {
        let x_vec1 = ThreeValCol::new(&vec![0, 0, 1, 2, 2, 1, 0, 1]);
        let x_vec2 = ThreeValCol::new(&vec![0, 1, 1, 1, 0, 1, 0, 1]);
        let mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6]);
        let oob_mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let mult1 = MultiX::ThreeVal(x_vec1);
        let mult2 = MultiX::ThreeVal(x_vec2);
        let x_df = XDf{data: vec!(mult1, mult2), idx_to_splitid_map: vec![0, 1], splitid_to_idx_map: vec![0, 1]};
        let idx = ColSplitIndex {col_id: 0, pivot: MultiPivot::ThreeVal(ThreeValPivot::NotRed), shadow: false};
        assert_eq!(x_df.make_split(idx,
                                   &mask,
                                   &RngFactory::new(1, Some(100), Some(100)),
                                   None),
                   [Mask::new(vec![2, 3, 4, 5]), Mask::new(vec![0, 1, 6])]);
    }

    #[test]
    fn shadowable_can_subset() {
        let x_vec1 = ThreeValCol::new(&vec![0, 0, 0]);
        let x_vec2 = ThreeValCol::new(&vec![0, 0, 1]);
        let x_vec3 = ThreeValCol::new(&vec![0, 1, 0]);
        let x_vec4 = ThreeValCol::new(&vec![1, 0, 0]);

        let mult1 = MultiX::ThreeVal(x_vec1);
        let mult2 = MultiX::ThreeVal(x_vec2);
        let mult3 = MultiX::ThreeVal(x_vec3);
        let mult4 = MultiX::ThreeVal(x_vec4);

        let x_df = XDf::new(vec![mult1, mult2, mult3.clone(), mult4.clone()]);
        let idxs: Vec<SplitColId> = (2..=3).rev().map(|i| SplitColId {col_id: i, shadow: false}).collect();
        let new_df = x_df.subset(&idxs);

        let expected_res = XDf {
            data: vec![mult4, mult3],
            idx_to_splitid_map: vec![3, 2],
            splitid_to_idx_map: vec![0, 1, 1, 0]
        };
        assert_eq!(new_df, expected_res);
    }

    #[test]
    fn shadowable_can_create_shadows() {
        let x_vec1 = ThreeValCol::new(&vec![0, 0, 0, 0]);
        let x_vec2 = ThreeValCol::new(&vec![0, 0, 1, 0]);
        let x_vec3 = ThreeValCol::new(&vec![0, 1, 0, 1]);
        let x_vec4 = ThreeValCol::new(&vec![1, 0, 1, 1]);
        let x_vec5 = ThreeValCol::new(&vec![1, 1, 1, 1]);

        let mult1 = MultiX::ThreeVal(x_vec1);
        let mult2 = MultiX::ThreeVal(x_vec2);
        let mult3 = MultiX::ThreeVal(x_vec3);
        let mult4 = MultiX::ThreeVal(x_vec4);
        let mult5 = MultiX::ThreeVal(x_vec5);

        let mut x_df = XDf::new(vec![mult1, mult2, mult3, mult4, mult5]);
        x_df.add_shadows(RngFactory::new(1, None, None));

        assert_eq!(x_df.get_ncol(), 10);
        assert_eq!(x_df.idx_to_splitid_map, (0..10).collect::<Vec<usize>>());
        assert_eq!(x_df.splitid_to_idx_map, (0..10).collect::<Vec<usize>>());
    }

    #[test]
    fn shadowable_can_create_shadows_with_less_than_five_attributes() {
        let x_vec1 = ThreeValCol::new(&vec![0, 0, 0]);
        let x_vec2 = ThreeValCol::new(&vec![0, 0, 1]);
        let x_vec3 = ThreeValCol::new(&vec![0, 1, 0]);
        let x_vec4 = ThreeValCol::new(&vec![1, 0, 0]);

        let mult1 = MultiX::ThreeVal(x_vec1);
        let mult2 = MultiX::ThreeVal(x_vec2);
        let mult3 = MultiX::ThreeVal(x_vec3);
        let mult4 = MultiX::ThreeVal(x_vec4);

        let x_df = XDf::new(vec![mult1, mult2, mult3, mult4]);
        let idxs: Vec<SplitColId> = (1..=2).rev().map(|i| SplitColId {col_id: i, shadow: false}).collect();

        let mut new_df = x_df.subset(&idxs);
        new_df.add_shadows(RngFactory::new(1, None, None));

        assert_eq!(new_df.get_ncol(), 10);
        assert_eq!(new_df.idx_to_splitid_map, [2, 1, 3, 4, 5, 6, 7, 8, 9, 10]);
    }
}
