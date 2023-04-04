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

use crate::random_number_generator::Rng;

#[derive(Clone)]
pub struct RngFactory {
    seed: u64,
    ncol: Option<usize>,
    ntree: Option<usize>
}

impl RngFactory {
    pub fn new(seed: u64, ncol: Option<usize>, ntree: Option<usize>) -> RngFactory {
        return RngFactory {seed, ncol, ntree};
    }

    #[inline]
    pub fn new_rng_shadow(&self, col_id: usize) -> Rng {
        return Rng::new(self.seed, (col_id + 1) as u64);
    }

    #[inline]
    pub fn new_rng_tree(&self, ith_tree: usize) -> Rng {
        return Rng::new(self.seed, (self.ncol.expect("No ncol provided") + ith_tree + 1) as u64);
    }

    #[inline]
    pub fn new_rng_tree_mask(&self, ith_tree: usize) -> Rng {
        let incr = self.ncol.expect("No ncol provided") +
            self.ntree.expect("No ntree provided") +
            ith_tree + 1;
        return Rng::new(self.seed, incr as u64);
    }

    #[inline]
    pub fn new_rng_permutation(&self, ith_tree: usize, col_id: usize) -> Rng {
        let ncol = self.ncol.expect("No ncol provided");
        let incr = ncol +
            self.ntree.expect("No ntree provided") * 2 +
            ith_tree*ncol + col_id + 1;
        return Rng::new(self.seed, incr as u64);
    }
}