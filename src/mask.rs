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

use std::collections::HashSet;
use crate::random_number_generator::Rng;

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Mask(Vec<usize>);

impl Mask {
    pub fn new(mask_val: Vec<usize>) -> Mask {
        let mut x = mask_val.clone();
        x.sort();
        Mask(x)
    }

    #[inline]
    pub fn random_mask(n: usize, sample_fraction: f64, rng: &mut Rng) -> Mask {
        let k = (n as f64 * sample_fraction).floor() as usize;
        let range: Vec<usize> = (0..n).collect();
        let mask_vec = rng.sample(range.as_slice(), k);
        return Mask::new(mask_vec);
    }

    #[inline]
    pub fn get_mask(&self) -> &Vec<usize> {
        return &self.0;
    }

    #[inline]
    pub fn get_by_mask<T: Copy>(&self, x: &[T]) -> Vec<T> {
        self.0.iter().map(|&i| x[i]).collect()
    }

    #[inline]
    pub fn inverse(&self, range: &[usize]) -> Mask {
        let range_set: HashSet<usize> = range.iter().cloned().collect();
        let mask_set: HashSet<usize> = self.get_mask().iter().cloned().collect();
        let new_mask: Vec<usize> = (&range_set - &mask_set).iter().cloned().collect();
        return Mask::new(new_mask);
    }

    #[inline]
    pub fn len(&self) -> usize {
        return self.0.len();
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use crate::mask::Mask;
    use crate::random_number_generator::Rng;

    #[test]
    fn inverse_mask() {
        let mask = Mask::new(vec![1,3]);
        let mut res = mask.inverse(&(0..5).collect::<Vec<usize>>()).get_mask().clone();
        res.sort();
        assert_eq!(res, vec![0, 2, 4]);
    }

    #[test]
    fn get_random_mask() {
        const N: usize = 100000;
        const FRAC: f64 = 0.5;

        (1..N).map(|i| { Mask::random_mask(6, FRAC, &mut Rng::new(i as u64, 1)) })
            .fold(vec![], |mut acc, a| {acc.extend(a.get_mask().clone()); acc}).iter()
            .fold([0; 6], |mut acc, &a| {
                assert!(a >= 0);
                assert!(a < N);
                acc[a] += 1;
                acc
            })
            .map(|x| {
                let p = x as f64/N as f64;
                assert_approx_eq!(f64, p, FRAC, epsilon=0.01)
            });
    }
}