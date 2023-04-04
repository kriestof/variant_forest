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

use std::cmp::Ordering;

use crate::mask::Mask;
use crate::data_interface::{Predicted, Response};
use crate::random_number_generator::Rng;

pub type Y = bool;

#[derive(Debug)]
pub struct YBool(Vec<Y>);

impl YBool {
    pub fn new(x: &[Y]) -> YBool {
        return YBool(x.clone().to_vec());
    }
}

impl Response<Y> for YBool {
    fn get_class(&self, mask: &Mask) -> Option<Y> {
        let acc = mask.get_by_mask(&self.0)
            .iter()
            .fold([false, false], |mut acc, &x| { acc[x as usize] = true; acc});

        return match (acc[0], acc[1]) {
            (true, true) => None,
            (true, false) => Some(false),
            (false, true) => Some(true),
            (false, false) => unreachable!()
        };
    }

    fn get_major_class(&self, mask: &Mask, rng: &mut Rng) -> Y {
        if mask.get_mask().len() == 0 {
            panic!("Cannot give major class for empty vector.");
        }

        let acc = mask.get_by_mask(&self.0)
            .iter()
            .fold((0, 0), |acc, &x| {
                match x {
                    false => (acc.0+1, acc.1),
                    true => (acc.0, acc.1+1)
                }
            });

        return match acc.0.cmp(&acc.1) {
            Ordering::Greater => false,
            Ordering::Less => true,
            Ordering::Equal => rng.rand_uni() > 0.5,
        }
    }

    #[inline]
    fn pred_incorrect(&self, mask: &Mask, preds: &Predicted<Y>) -> u64 {
        mask.get_by_mask(&self.0).iter().zip(preds.iter()).fold(0, |mut acc, x| {
            if x.0 != x.1 {
                acc += 1;
            }
            acc
        })
    }

    fn pred_error(&self, mask: &Mask, preds: &Predicted<Y>) -> f64 {
        return self.pred_incorrect(&mask, &preds) as f64/preds.len() as f64;
    }

    #[inline]
    fn as_vector(&self) -> Vec<Y> {
        return self.0.clone();
    }

    #[inline]
    fn as_vector_ref(&self) -> &Vec<Y> {
        return &self.0;
    }

    #[inline]
    fn len(&self) -> usize {
        return self.0.len();
    }
}

#[cfg(test)]
mod tests {
    use crate::mask::Mask;
    use crate::data_interface::Response;
    use crate::data_interface::y_bool::YBool;
    use crate::random_number_generator::Rng;

    #[test]
    fn pred_incorrect_returns_correct_value() {
        let y = YBool(vec![true, true, true, false, false, false, false, false]);
        let mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let preds = vec![true, true, true, false, false, false, false, true];
        assert_eq!(y.pred_incorrect(&mask, &preds), 1);
    }

    #[test]
    fn pred_incorrect_returns_correct_value_all_correct() {
        let y = YBool(vec![true, true, true, false, false, false, false, false]);
        let mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let preds = vec![true, true, true, false, false, false, false, false];
        assert_eq!(y.pred_incorrect(&mask, &preds), 0);
    }

    #[test]
    fn pred_error_returns_correct_value() {
        let y = YBool(vec![true, true, true, false, false, false, false, false]);
        let mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let preds = vec![true, true, true, false, false, false, false, true];
        assert_eq!(y.pred_error(&mask, &preds), 1./8.);
    }

    #[test]
    fn pred_error_returns_correct_value_all_correct() {
        let y = YBool(vec![true, true, true, false, false, false, false, false]);
        let mask = Mask::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let preds = vec![true, true, true, false, false, false, false, false];
        assert_eq!(y.pred_error(&mask, &preds), 0.);
    }

    #[test]
    fn get_class_returns_none_with_many_classes() {
        let y = YBool(vec![true, true, false, true]);
        let mask = Mask::new(vec![0, 1, 2, 3]);
        assert!(y.get_class(&mask).is_none());
    }

    #[test]
    fn get_class_returns_false_class() {
        let y = YBool(vec![true, true, false, true]);
        let mask = Mask::new(vec![2]);
        assert_eq!(y.get_class(&mask), Some(false));
    }

    #[test]
    fn get_class_returns_true_class() {
        let y = YBool(vec![true, true, false, true]);
        let mask = Mask::new(vec![0, 1, 3]);
        assert_eq!(y.get_class(&mask), Some(true));
    }

    #[test]
    fn get_class_should_panic() {
        let y = YBool(vec![true, true, false, true]);
        let mask = Mask::new(vec![]);
        let res = std::panic::catch_unwind(|| {
            y.get_class(&mask);
        });
        assert!(res.is_err());
    }

    #[test]
    fn get_major_class() {
        let mut rng = Rng::new(0, 1);
        let y = YBool(vec![true, true, false, true]);
        let mask = Mask::new(vec![0, 1, 2, 3]);
        assert_eq!(y.get_major_class(&mask, &mut rng), true);

        let y = YBool(vec![true, false, false, false]);
        let mask = Mask::new(vec![0, 1, 2, 3]);
        assert_eq!(y.get_major_class(&mask, &mut rng), false);

        let y = YBool(vec![true, true, false, false]);
        let mask = Mask::new(vec![0, 1, 2, 3]);
        assert_eq!(y.get_major_class(&mask, &mut rng), true);
    }

    #[test]
    #[should_panic(expected = "Cannot give major class for empty vector.")]
    fn get_major_class_empty_vector_should_panic() {
        let mut rng = Rng::new(0, 1);
        let y = YBool(vec![true, true, false, true]);
        let mask = Mask::new(vec![]);

        y.get_major_class(&mask, &mut rng);
    }
}