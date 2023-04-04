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

use crate::data_interface::three_val::{ThreeValCol, ThreeValOpt};
use crate::data_interface::three_val::ThreeVal::*;

#[inline]
fn _gini(num_xt_yt: usize, num_xt_yf: usize, num_xf_yt: usize, num_xf_yf: usize, n: f64) -> f64 {
    let mut res = 0.;
    let (np_xt_yt, np_xt_yf, np_xf_yt, np_xf_yf) = (num_xt_yt as f64, num_xt_yf as f64, num_xf_yt as f64, num_xf_yf as f64);

    if num_xt_yt + num_xt_yf > 0 {
        res += (np_xt_yt + np_xt_yf) / n - (np_xt_yt * np_xt_yt + np_xt_yf * np_xt_yf ) / (np_xt_yt + np_xt_yf) / n;
    }

    if num_xf_yt + num_xf_yf > 0 {
        res += (np_xf_yt + np_xf_yf) / n - (np_xf_yt * np_xf_yt + np_xf_yf * np_xf_yf ) / (np_xf_yt + np_xf_yf) / n;
    }

    return res;
}

#[inline]
pub fn gini_x_threeval_y_bool<'a, Ix, Iy>(x: &mut Ix, y: &mut Iy, n: usize) -> (f64, f64, f64)
where
    Ix: Iterator<Item=ThreeValOpt>,
    Iy: Iterator<Item=bool>
{
    if n == 0 {
        panic!("Empty vectors given.");
    }

    let [num_xr_yt, num_xr_yf, num_xg_yt, num_xg_yf, num_xb_yt, num_xb_yf]: [usize; 6] = x
        .zip(y)
        .fold([0, 0, 0, 0, 0, 0], |mut c, (x, y)| {
            match (x.unwrap(), y) {
                (Red, true) => c[0] += 1,
                (Red, false) => c[1] += 1,
                (Green, true) => c[2] += 1,
                (Green, false) => c[3] += 1,
                (Blue, true) => c[4] += 1,
                (Blue, false) => c[5] += 1,
            }
            c
        });


    let nf = n as f64;
    let s = (
        _gini(num_xg_yt+num_xb_yt, num_xg_yf+num_xb_yf, num_xr_yt, num_xr_yf, nf),
        _gini(num_xr_yt+num_xb_yt, num_xr_yf+num_xb_yf, num_xg_yt, num_xg_yf, nf),
        _gini(num_xr_yt+num_xg_yt, num_xr_yf+num_xg_yf, num_xb_yt, num_xb_yf, nf)
    );

    return s;
}

#[cfg(test)]
mod tests {
    use super::{_gini, gini_x_threeval_y_bool};
    use float_cmp::assert_approx_eq;
    use crate::data_interface::three_val::{ThreeVal, ThreeValOpt};

    #[test]
    fn part_gini_calculated_correctly() {
        let p1 = 3./9. - (1 + 2*2) as f64/3./9.;
        let p2 = 6./9. - (4*4 + 2*2) as f64/6./9.;
        assert_approx_eq!(f64, _gini(1, 2, 4, 2, 9.), p1+p2);
    }

    #[test]
    fn part_gini_can_handle_single_x_class() {
        assert_approx_eq!(f64, _gini(2, 2, 0, 0, 4.), 0.5);
        assert_approx_eq!(f64, _gini(0, 0, 2, 2, 4.), 0.5)
    }

    #[test]
    fn part_gini_can_handle_single_y_class() {
        assert_approx_eq!(f64, _gini(2, 0, 2, 0, 4.), 0.);
        assert_approx_eq!(f64, _gini(0, 2, 0, 2, 4.), 0.);
    }


    #[test]
    fn gini_should_panic_with_empty_vectors() {
        let x: Vec<ThreeValOpt> = vec![];
        let y: Vec<bool> = vec![];

        let res = std::panic::catch_unwind(|| {
            gini_x_threeval_y_bool(&mut x.into_iter(), &mut y.into_iter(), 0);
        });
        assert!(res.is_err());
    }

    #[test]
    fn gini_calculated_correctly() {
        let x: Vec<ThreeValOpt> = vec![0, 2, 2, 1, 1, 0, 2, 0, 1].iter().map(|&x| Some(match x {
            0 => ThreeVal::Red,
            1 => ThreeVal::Green,
            2 => ThreeVal::Blue,
            _ => panic!("Out of enum bounds")
        })).collect();

        let y = vec![false, true, true, false, true, false, true, true, false];

        let res = gini_x_threeval_y_bool(&mut x.into_iter(), &mut y.into_iter(), 9);
        assert_approx_eq!(f64, res.0, 3./9. - (1.+2.*2.)/3./9. + 6./9. - (2.*2.+4.*4.)/6./9.);
        assert_approx_eq!(f64, res.1, 3./9. - (1.+2.*2.)/3./9. + 6./9. - (2.*2.+4.*4.)/6./9.);
        assert_approx_eq!(f64, res.2, 6./9. - (4*4+2*2) as f64/6./9.);
    }
}
