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

use std::hash::Hash;
use std::fmt::Debug;
use crate::data_interface::y_bool::Y;

use crate::mask::Mask;
use crate::random_number_generator::factory::RngFactory;
use crate::random_number_generator::Rng;

pub mod multi_x;
pub mod three_val;
pub mod y_bool;

pub type Predicted<T> = Vec<T>;

pub trait Permutable {
    fn permute(&self, perm_rng: Rng, oob_mask: &Mask) -> Self;
}

pub trait Splittable<Y>: Permutable {
    type Pivot;
    fn split_with_pivot(&self, mask: &Mask, p: &Self::Pivot, shadow_rng: Option<Rng>) -> [Mask; 2];
    fn gen_optimal_pivot<T>(&self, mask: &Mask, y:  &T, shadow_rng: Option<Rng>) -> (Self::Pivot, f64)
    where
        T: Response<Y>;
}

pub trait Response<T> {
    fn pred_incorrect(&self, mask: &Mask, preds: &Predicted<T>) -> u64;
    fn get_class(&self, mask: &Mask) -> Option<T>;
    fn get_major_class(&self, mask: &Mask, rng: &mut Rng) -> T;
    fn pred_error(&self, mask: &Mask, preds: &Predicted<T>) -> f64;
    fn as_vector(&self) -> Vec<T>;
    fn as_vector_ref(&self) -> &Vec<T>;
    fn len(&self) -> usize;
}

pub trait Shadowable<Split, Y>:  DataInterface<Split, Y> where
    Split: ColumnIdentifiable
{
    fn subset(&self, idxs: &Vec<Split::Col>) -> Self;
    fn add_shadows(&mut self, rng_factory: RngFactory);
    fn get_col_ids(&self) -> Vec<Split::Col>;
}

pub trait DataInterface<Split, Y> where
    Split: ColumnIdentifiable
{
    type InternalType;

    fn get_ncol(&self) -> usize;
    fn find_min_idx<T>(&self, mask: &Mask, y:  &T, mtry: usize, rng: &mut Rng, rng_factory: &RngFactory, shadow_vars: bool) -> Split
    where
        T: Response<Y>;
    fn make_split(&self, idx: Split, mask: &Mask, rng_factory: &RngFactory, permuted_vec: Option<&Self::InternalType>) -> [Mask; 2];
    fn permute_index(&self, idx: Split::Col, rng_factory: &RngFactory, oob_mask: &Mask, ith_tree: usize) -> Self::InternalType;
}

pub trait ColumnIdentifiable {
    type Col: Hash + Eq + Copy + Send;
    fn get_col_id(&self) -> Self::Col;
}
