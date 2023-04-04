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

pub mod factory;

const MULTIPLIER: u64 = 6364136223846793005;
const ROTATE: u32 = 59;
const XSHIFT: u32 = 18;
const SPARE: u32 = 27;

const SALT: u64 =  match cfg!(test) {
    false => 77585124950184,
    true => 0
};

#[derive(Clone, Copy)]
pub struct Rng {
    state: u64,
    increment: u64,
}

//PCG 32 based random number generator with increment=1
//Based on Melissa E. O'Neill. PCG: A Family of Simple Fast Space-Efficient Statistically Good
//Algorithms for Random Number Generation. Harvey Mudd College, 2014.
//Implementation inspired by rust-random/rand pcg32 generator and mbq/wybr package
impl Rng {
    pub fn new(seed: u64, increment: u64) -> Self {
        if increment == 0 {
            panic!("Increment must be larger than 0.");
        }

        let mut pcg = Rng {state: seed+SALT, increment: increment};
        pcg.state = pcg.state.wrapping_add(pcg.increment);
        pcg.step();
        pcg
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULTIPLIER).wrapping_add(self.increment);
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let state = self.state;
        self.step();

        let rot = (state >> ROTATE) as u32;
        let xsh = (((state >> XSHIFT) ^ state) >> SPARE) as u32;
        xsh.rotate_right(rot)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut res = self.next_u32() as u64;
        res <<= 32;
        res += self.next_u32() as u64;
        res
    }

    // This is from oorandom package
    // It's possible to use u64. Though, next_u64 should be better implemented and next_usize must be adjusted.
    #[inline]
    fn next_usize(&mut self, up_to: u32) -> usize {
        assert!(std::mem::size_of::<usize>() <= 64);
        assert!(up_to > 0);

        if up_to == 1 {
            return 0;
        }

        let mut m: u64 = u64::from(self.next_u32()) * u64::from(up_to);
        let mut leftover: u32 = (m & 0xFFFF_FFFF) as u32;

        if leftover < up_to {
            // TODO: verify the wrapping_neg() here
            let threshold: u32 = up_to.wrapping_neg() % up_to;
            while leftover < threshold {
                m = u64::from(self.next_u32()).wrapping_mul(u64::from(up_to));
                leftover = (m & 0xFFFF_FFFF) as u32;
            }
        }
        (m >> 32) as usize
    }

    pub fn shuffle<T: Clone + Copy>(&mut self, x: &mut [T]) {
        if !x.is_empty() {
            for e in 0..(x.len() - 1) {
                let ee = e + self.next_usize((x.len() - e) as u32);
                x.swap(e, ee);
            }
        }
    }

    #[inline]
    pub fn rand_uni(&mut self) -> f64 {
        loop {
            let res = self.next_u64() as f64/u64::MAX as f64;
            if res != 0. {
                break res;
            }
        }
    }

    // Reservoir sampling algorithm L
    pub fn sample<T: Copy>(&mut self, x: &[T], k: usize) -> Vec<T> {
        if k > x.len() {
            panic!("Cannot sample when k is greater than n.");
        }

        let mut res: Vec<_> = (0..k).collect();
        let mut w = (self.rand_uni().ln()/k as f64).exp();
        let mut i = k-1;
        let n = x.len();

        while i <= n {
            i += (self.rand_uni().ln()/(1.-w).ln()).floor() as usize + 1;
            if i < n {
               res[self.next_usize(k as u32)] = i;
               w *= (self.rand_uni().ln()/k as f64).exp();
            }
        }

        res.iter().map(|&i| x[i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use super::*;

    #[test]
    fn cmp_with_c() {
        let mut x = Rng::new(21, 1);
        let x_out: Vec<u32> = (0..6).map(|_| x.next_u32()).collect();
        //For seed=21 & stream "0" (streams are *2+1 because stream number must be odd)
        let pcg32_c_out = vec![
            4046551126, 3645130801, 1491492233, 2234036793, 669229171, 981735442,
        ];
        assert_eq!(x_out, pcg32_c_out);

        let mut x = Rng::new(33198495, 1);
        let x_out: Vec<u32> = (0..6).map(|_| x.next_u32()).collect();
        //For seed=33198495 & stream "0" (streams are *2+1 because stream number must be odd)
        let pcg32_c_out = vec![
            3221354117, 560577576, 1464394025, 77867303, 6303390, 439113613,
        ];
        assert_eq!(x_out, pcg32_c_out);
    }

    #[test]
    fn upto() {
        let mut x = Rng::new(912, 1);
        let n = 3;
        let mut cc: Vec<u32> = std::iter::repeat(0).take(n).collect();
        for _ in 0..100 {
            cc[x.next_usize(n as u32)] += 1;
        }
        assert_eq!(cc[1], 32); //Will change with seed
    }

    #[test]
    fn shuffle() {
        let mut x = Rng::new(81, 1);
        let mut v: Vec<u32> = vec![];
        x.shuffle(&mut v);
        v.push(1);
        x.shuffle(&mut v);
        v.push(2);
        x.shuffle(&mut v);
        assert_eq!(v, vec![1, 2]);
    }

    #[test]
    fn rand_uni() {
        let mut rng = Rng::new(122, 1);
        let max = (1..10000).map(|_| rng.rand_uni()).max_by(|a, b| a.total_cmp(b));
        let min = (1..10000).map(|_| rng.rand_uni()).min_by(|a, b| a.total_cmp(b));
        let mean = (1..100_000).map(|_| rng.rand_uni()).sum::<f64>()/100_000.;
        assert_approx_eq!(f64, max.unwrap(), 1., epsilon=0.001);
        assert_approx_eq!(f64, min.unwrap(), 0., epsilon=0.001);
        assert_approx_eq!(f64, mean, 0.5, epsilon=0.001);
    }

    #[test]
    fn sample_k_1() {
        let x: [usize; 4] = [0, 1, 2, 3];
        let res: Vec<_> = (1..100_000).map(|i| {
            let mut rng = Rng::new(i, 1);
            rng.sample(&x, 1)[0]
        }).fold(vec![0, 0, 0, 0], |mut acc, a| {acc[a] += 1; acc}).into_iter().map(|a| a as f64/100_000.).collect();
        assert_approx_eq!(f64, res[0], 0.25, epsilon=0.01);
        assert_approx_eq!(f64, res[1], 0.25, epsilon=0.01);
        assert_approx_eq!(f64, res[2], 0.25, epsilon=0.01);
        assert_approx_eq!(f64, res[3], 0.25, epsilon=0.01);
    }

    #[test]
    fn sample_k_eq_n() {
        let mut rng = Rng::new(7, 1);
        let x = ["A", "B", "C", "D", "E", "F", "G"];
        assert_eq!(rng.sample(&x, 7), &x);
    }
}