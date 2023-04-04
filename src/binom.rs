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

use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::f64;


/// Constant value for `ln(pi)`
pub const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;


/// Constant value for `ln(2 * sqrt(e / pi))`
pub const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;

/// Standard epsilon, maximum relative precision of IEEE 754 double-precision
/// floating point numbers (64 bit) e.g. `2^-53`
pub const F64_PREC: f64 = 0.00000000000000011102230246251565;

/// Auxiliary variable when evaluating the `gamma_ln` function
const GAMMA_R: f64 = 10.900511;

/// Polynomial coefficients for approximating the `gamma_ln` function
const GAMMA_DK: &[f64] = &[
    2.48574089138753565546e-5,
    1.05142378581721974210,
    -3.45687097222016235469,
    4.51227709466894823700,
    -2.98285225323576655721,
    1.05639711577126713077,
    -1.95428773191645869583e-1,
    1.70970543404441224307e-2,
    -5.71926117404305781283e-4,
    4.63399473359905636708e-6,
    -2.71994908488607703910e-9,
];


// function to silence clippy on the special case when comparing to zero.
#[inline(always)]
fn is_zero(x: f64) -> bool {
    x == 0.
}

pub fn binom_cdf(k: u64, n: u64, p: f64) -> f64 {
    if k >= n {
        1.0
    } else {
        checked_beta_reg((n - k) as f64, k as f64 + 1.0, 1.0 - p)
    }
}


/// Computes the logarithm of the gamma function
/// with an accuracy of 16 floating point digits.
/// The implementation is derived from
/// "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

        LN_PI
            - (f64::consts::PI * x).sin().ln()
            - s.ln()
            - LN_2_SQRT_E_OVER_PI
            - (0.5 - x) * ((0.5 - x + GAMMA_R) / f64::consts::E).ln()
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

        s.ln()
            + LN_2_SQRT_E_OVER_PI
            + (x - 0.5) * ((x - 0.5 + GAMMA_R) / f64::consts::E).ln()
    }
}

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Errors
///
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn checked_beta_reg(a: f64, b: f64, x: f64) -> f64 {
        let bt = if is_zero(x) || x == 1. {
            0.0
        } else {
            (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
                + a * x.ln()
                + b * (1.0 - x).ln())
                .exp()
        };
        let symm_transform = x >= (a + 1.0) / (a + b + 2.0);
        let eps = F64_PREC;
        let fpmin = f64::MIN_POSITIVE / eps;

        let mut a = a;
        let mut b = b;
        let mut x = x;
        if symm_transform {
            let swap = a;
            x = 1.0 - x;
            a = b;
            b = swap;
        }

        let qab = a + b;
        let qap = a + 1.0;
        let qam = a - 1.0;
        let mut c = 1.0;
        let mut d = 1.0 - qab * x / qap;

        if d.abs() < fpmin {
            d = fpmin;
        }
        d = 1.0 / d;
        let mut h = d;

        for m in 1..141 {
            let m = f64::from(m);
            let m2 = m * 2.0;
            let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;

            if d.abs() < fpmin {
                d = fpmin;
            }

            c = 1.0 + aa / c;
            if c.abs() < fpmin {
                c = fpmin;
            }

            d = 1.0 / d;
            h = h * d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;

            if d.abs() < fpmin {
                d = fpmin;
            }

            c = 1.0 + aa / c;

            if c.abs() < fpmin {
                c = fpmin;
            }

            d = 1.0 / d;
            let del = d * c;
            h *= del;

            if (del - 1.0).abs() <= eps {
                return if symm_transform {
                    return 1.0 - bt * h / a;
                } else {
                    return bt * h / a;
                };
            }
        }

        if symm_transform {
            return 1.0 - bt * h / a;
        } else {
            return bt * h / a;
        }
}


#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use crate::binom::{checked_beta_reg, binom_cdf};

    #[test]
    fn beta_cdf_calculated_correctly() {
        assert_approx_eq!(f64, checked_beta_reg(2., 2., 0.5), 0.5, epsilon=0.000001);
        assert_approx_eq!(f64, checked_beta_reg(2., 2., 0.05), 0.00725, epsilon=0.000001);
        assert_approx_eq!(f64, checked_beta_reg(2., 2., 0.95), 0.99275, epsilon=0.000001);
    }

    #[test]
    fn binom_cdf_calculated_correctly() {
        assert_approx_eq!(f64, binom_cdf(0, 10, 0.5), 0.0009765625, epsilon=0.000001);
        assert_approx_eq!(f64, binom_cdf(1, 10, 0.5), 0.01074219, epsilon=0.000001);
        assert_approx_eq!(f64, binom_cdf(9, 10, 0.5), 0.9990234, epsilon=0.000001);
        assert_approx_eq!(f64, binom_cdf(10, 10, 0.5), 1., epsilon=0.000001);
    }
}