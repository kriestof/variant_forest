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

#[inline]
pub fn gini_x_bool_y_bool(x: &Vec<bool>, y: &Vec<bool>) -> f64 {
    if x.len() != y.len() {
        panic!("X & Y size mismatch!");
    }

    if x.len() == 0 {
        panic!("Empty vectors given.");
    }

    let [num_xt_yt, num_xt_yf, num_xf_yt, num_xf_yf]: [usize; 4] = x
        .iter()
        .zip(y.iter())
        .fold([0, 0, 0, 0], |mut c, (&x, &y)| {
            match (x, y) {
                (true, true) => c[0] += 1,
                (true, false) => c[1] += 1,
                (false, true) => c[2] += 1,
                (false, false) => c[3] += 1
            }
            c
        });

    let n = x.len() as f64;
    let (np_xt_yt, np_xt_yf, np_xf_yt, np_xf_yf) = (num_xt_yt as f64, num_xt_yf as f64, num_xf_yt as f64, num_xf_yf as f64);
    let mut res = 0.;

    if num_xt_yt + num_xt_yf > 0 {
        res += (np_xt_yt + np_xt_yf) / n - (np_xt_yt * np_xt_yt + np_xt_yf * np_xt_yf ) / (np_xt_yt + np_xt_yf) / n;
    }

    if num_xf_yt + num_xf_yf > 0 {
        res += (np_xf_yt + np_xf_yf) / n - (np_xf_yt * np_xf_yt + np_xf_yf * np_xf_yf ) / (np_xf_yt + np_xf_yf) / n;
    }

    return res;
}

#[cfg(test)]
mod tests {
    use super::gini_x_bool_y_bool;
    use float_cmp::assert_approx_eq;

    #[test]
    fn gini_calculated_correctly() {
        let x = vec![true, false, false, false, false, true, false, true, false];
        let y = vec![false, true, true, false, true, false, true, true, false];

        let p1 = 3./9. - (1 + 2*2) as f64/3./9.;
        let p2 = 6./9. - (4*4 + 2*2) as f64/6./9.;
        assert_approx_eq!(f64, gini_x_bool_y_bool(&x, &y), p1+p2);
    }

    #[test]
    fn gini_can_handle_single_x_class() {
        let x = vec![true, true, true, true];
        let y = vec![true, true, false, false];

        assert_approx_eq!(f64, gini_x_bool_y_bool(&x, &y), 0.5);

        let x = vec![false, false, false, false];
        let y = vec![true, true, false, false];

        assert_approx_eq!(f64, gini_x_bool_y_bool(&x, &y), 0.5)
    }

    #[test]
    fn gini_can_handle_single_y_class() {
        let x = vec![true, true, false, false];
        let y = vec![true, true, true, true];

        assert_approx_eq!(f64, gini_x_bool_y_bool(&x, &y), 0.);

        let x = vec![true, true, false, false];
        let y = vec![false, false, false, false];

        assert_approx_eq!(f64, gini_x_bool_y_bool(&x, &y), 0.);
    }

    #[test]
    fn gini_should_panic_with_different_vector_sizes() {
        let x = vec![true, true, false, false, false];
        let y = vec![true, true, true, true];

        let res = std::panic::catch_unwind(|| {
            gini_x_bool_y_bool(&x, &y);
        });
        assert!(res.is_err());
    }

    #[test]
    fn gini_should_panic_with_empty_vectors() {
        let x = vec![];
        let y = vec![];

        let res = std::panic::catch_unwind(|| {
            gini_x_bool_y_bool(&x, &y);
        });
        assert!(res.is_err());
    }
}