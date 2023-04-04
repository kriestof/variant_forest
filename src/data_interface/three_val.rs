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
use crate::data_interface::{Permutable, Response, Splittable};
use crate::data_interface::y_bool::Y;
use crate::gini::x_bool_y_bool::gini_x_bool_y_bool;
use crate::gini::x_threeval_y_bool::gini_x_threeval_y_bool;
use crate::random_number_generator::Rng;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ThreeVal {
    Red,
    Green,
    Blue,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ThreeValPivot {
    NotRed,
    NotGreen,
    NotBlue,
}

pub type ThreeValOpt = Option<ThreeVal>;
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ThreeValCol(Vec<ThreeValOpt>);


impl ThreeValCol {
    pub fn new(arr: &[i8]) -> Self{
        return Self(arr.iter().map(|&x| Some(match x {
            0 => ThreeVal::Red,
            1 => ThreeVal::Green,
            2 => ThreeVal::Blue,
            _ => panic!("Out of enum bounds")
        })).collect());
    }

    pub fn len(&self) -> usize {
        return self.0.len();
    }
}

impl Permutable for ThreeValCol {
    fn permute(&self, mut perm_rng: Rng, oob_mask: &Mask) -> ThreeValCol {
        let mut x = oob_mask.get_by_mask(&self.0.clone());
        perm_rng.shuffle(&mut x);

        let mut x_full = self.0.clone();
        for (&xv, &i) in x.iter().zip(oob_mask.get_mask().iter()) {
            x_full[i] = xv;
        }

        return ThreeValCol(x_full);
    }
}

impl Splittable<Y> for ThreeValCol {
    type Pivot = ThreeValPivot;

    fn split_with_pivot(&self, mask: &Mask, p: &Self::Pivot, shadow_rng: Option<Rng>) -> [Mask; 2] {
        // let mut x: Vec<_> = self.0.clone();
        let x = mask.get_by_mask(&self.0);
        // if shadow_rng.is_some() {
        //     shadow_rng.unwrap().shuffle(&mut x);
        // } // TODO remove shadows


        return x.iter().zip(mask.get_mask().iter()).fold([Vec::new(), Vec::new()], |mut acc, row| {
            if *p == row.0.unwrap() {
                acc[0].push(*row.1)
            } else {
                acc[1].push(*row.1)
            }
            acc
        }).map(|x| Mask::new(x))
    }

    fn gen_optimal_pivot<T>(&self, mask: &Mask, y: &T, shadow_rng: Option<Rng>) -> (Self::Pivot, f64)
    where
        T: Response<Y>
    {
        use ThreeValPivot::*;
        let x;
        let mut x_temp;

        if shadow_rng.is_some() {
            x_temp = self.0.clone();
            shadow_rng.unwrap().shuffle(&mut x_temp);
            x = &x_temp;
        } else {
            x = &self.0;
        }

        let mut x_fl = mask.get_mask().iter().map(|&i| x[i]);
        let y_vec = y.as_vector_ref();
        let mut y_fl = mask.get_mask().iter().map(|&i| y_vec[i]);

        let s = gini_x_threeval_y_bool(&mut x_fl, &mut y_fl, mask.len());

        // Yo, partial sort net (;
        return match (s.0 < s.1, s.0 < s.2, s.1 < s.2) {
            (true, true, _) => (NotRed, s.0),
            (false, _, true) => (NotGreen, s.1),
            (_, false, false) => (NotBlue, s.2),
            _ => unreachable!(),
        };
    }
}

impl PartialEq<ThreeVal> for ThreeValPivot {
    #[inline]
    fn eq(&self, other: &ThreeVal) -> bool {
        use ThreeVal::*;
        use ThreeValPivot::*;
        match (*self, *other) {
            (NotRed, Red) => false,
            (NotGreen, Green) => false,
            (NotBlue, Blue) => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use crate::mask::Mask;
    use crate::data_interface::three_val::{ThreeVal, ThreeValCol, ThreeValPivot};
    use crate::data_interface::{Permutable, Splittable};
    use crate::data_interface::y_bool::YBool;
    use crate::random_number_generator::Rng;

    #[test]
    fn make_split() {
        let x_vec = ThreeValCol(vec![0, 0, 1, 2, 2, 1, 0, 1].iter().map(|&x| Some(match x {
            0 => ThreeVal::Red,
            1 => ThreeVal::Green,
            2 => ThreeVal::Blue,
            _ => panic!("Out of enum bounds")
        })).collect());
        let mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6]);
        let oob_mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(x_vec.split_with_pivot(&mask, &ThreeValPivot::NotRed, None),
                   [Mask::new(vec![2, 3, 4, 5]), Mask::new(vec![0, 1, 6])]);
    }

    // #[test]
    // fn it_permutes() {
    //     let x_vec = ThreeValCol::new(&vec![0, 0, 1, 2, 2, 1, 0, 1, 0]);
    //     let mask = Mask::new(vec![1, 2, 3, 4, 5, 6, 7]);
    //     let res = x_vec.permute(Rng::new(1, 1), &mask);
    //     assert_eq!(res, ThreeValCol::new(&vec![0, 0, 1, 1, 2, 0, 2, 1, 0]))
    // }

    // #[test]
    // fn make_split_perm_seed_shadow() {
    //     let x_vec = ThreeValCol::new(&vec![0, 0, 1, 2, 2, 1, 0, 1, 0]);
    //     let mask = Mask::new(vec![1, 2, 3, 4, 5, 6, 7]);
    //     let oob_mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
    //     assert_eq!(x_vec.split_with_pivot(&mask,
    //                                       &ThreeValPivot::NotRed,
    //                                       Some(Rng::new(1, 1))),
    //                [Mask::new(vec![1, 4, 5, 7]), Mask::new(vec![2, 3, 6])]);
    // }

    // #[test]
    // fn make_split_perm_seed() {
    //     let x_vec = ThreeValCol(vec![0, 0, 1, 2, 2, 1, 0, 1, 0].iter().map(|&x| Some(match x {
    //         0 => ThreeVal::Red,
    //         1 => ThreeVal::Green,
    //         2 => ThreeVal::Blue,
    //         _ => panic!("Out of enum bounds")
    //     })).collect());
    //     let mask = Mask::new(vec![1, 2, 3, 4, 5, 6, 7]);
    //     let oob_mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
    //     assert_eq!(x_vec.split_with_pivot(&mask,
    //                                       &ThreeValPivot::NotRed,
    //                                       Some(Rng::new(1, 1)),
    //                                       None,
    //                                       &oob_mask),
    //                [Mask::new(vec![1, 3, 6, 7]), Mask::new(vec![2, 4, 5])]);
    // }

    #[test]
    fn gen_optimal_pivot() {
        let x = ThreeValCol::new(&vec![0, 2, 2, 1, 1, 0, 2, 0, 1]);
        let y = YBool::new(&vec![false, true, true, false, true, false, true, true, false]);
        let (piv, score) = x.gen_optimal_pivot(&Mask::new((0..=8).collect()), &y, None);
        assert_eq!(piv, ThreeValPivot::NotBlue);
        assert_approx_eq!(f64, score, 6./9. - (4*4+2*2) as f64/6./9.)
    }

    #[test]
    fn gen_optimal_pivot_uses_mask() {
        let x = ThreeValCol::new(&vec![1, 0, 2, 2, 1, 1, 0, 2, 0, 1, 1, 1]);
        let y = YBool::new(&vec![false, false, true, true, false, true, false, true, true, false, false, false]);
        let (piv, score) = x.gen_optimal_pivot(&Mask::new((1..=9).collect()), &y, None);
        assert_eq!(piv, ThreeValPivot::NotBlue);
        assert_approx_eq!(f64, score, 6./9. - (4*4+2*2) as f64/6./9.)
    }

    // #[test]
    // fn gen_optimal_pivot_filters_none() {
    //     let mut x = ThreeValCol::new(&vec![0, 2, 2, 1, 1, 0, 2, 0, 1]);
    //     x.0.insert(0, None);
    //     let y = YBool::new(&vec![false, false, true, true, false, true, false, true, true, false]);
    //     let (piv, score) = x.gen_optimal_pivot(&Mask::new((0..=9).collect()), &y, None);
    //     assert_eq!(piv, ThreeValPivot::NotBlue);
    //     assert_approx_eq!(f64, score, 6./9. - (4*4+2*2) as f64/6./9.)
    // }

    // #[test]
    // fn gen_optimal_pivot_perm_seed_shadow() {
    //     let x = ThreeValCol::new(&vec![1, 0, 2, 2, 1, 1, 0, 2, 0, 1, 1, 1]);
    //     let y = YBool::new(&vec![false, false, true, true, false, true, false, true, true, false, false, false]);
    //     let (piv, score) = x.gen_optimal_pivot(
    //         &Mask::new((1..=9).collect()),
    //         &y,
    //         Some(Rng::new(1, 1))
    //     );
    //     assert_eq!(piv, ThreeValPivot::NotRed);
    //     assert_approx_eq!(f64, score, 6./9. - (2*2+4*4) as f64/6./9. + 3./9. - (1+2*2) as f64/3./9.)
    // }
}
