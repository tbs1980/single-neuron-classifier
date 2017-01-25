extern crate rulinalg;
extern crate rand;

use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use rand::distributions::{Normal, IndependentSample};

fn sigmoid(v: f64) ->f64 {
    1.0_f64 / (1.0_f64 + (-v).exp())
}

fn main() {
    // create two clusters of data
    // http://briandolhansky.com/blog/2013/7/11/artificial-neural-networks-linear-classification-part-2
    let num_data_points: usize = 200;
    let num_dim: usize = 2;
    let mut data_x = Matrix::<f64>::zeros(num_data_points, num_dim);
    let mut class_t = Vector::<f64>::zeros(num_data_points);

    let normal = Normal::new(0.0, 0.3);

    for ind_i in 0..num_data_points {
        for ind_j in 0..num_dim {
            let v = normal.ind_sample(&mut rand::thread_rng());
            if ind_i < num_data_points / 2 {
                data_x[[ind_i,ind_j]] = v + 1.0_f64;
                class_t[ind_i] = -1.0_f64
            } else {
                data_x[[ind_i,ind_j]] = v + 3.0_f64;
                class_t[ind_i] = 1.0_f64
            }
        }
        println!("{} {} {} {}", ind_i, data_x[[ind_i, 0]], data_x[[ind_i, 1]], class_t[ind_i])
    }

    // run gradeint descent 

}
