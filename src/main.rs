extern crate rulinalg;
extern crate rand;

// use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::BaseMatrixMut;
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
    let mut data_x = Matrix::<f64>::zeros(num_data_points, num_dim+1);
    let mut class_t = Vector::<f64>::zeros(num_data_points);

    let normal = Normal::new(0.0, 0.3);

    for ind_i in 0..num_data_points {
        data_x[[ind_i,0]] = 1.0_f64;
        for ind_j in 1..num_dim+1 {
            let v = normal.ind_sample(&mut rand::thread_rng());
            if ind_i < num_data_points / 2 {
                data_x[[ind_i,ind_j]] = v + 1.0_f64;
                class_t[ind_i] = -1.0_f64
            } else {
                data_x[[ind_i,ind_j]] = v + 3.0_f64;
                class_t[ind_i] = 1.0_f64
            }
        }
        println!("{} {} {} {} {}", ind_i, data_x[[ind_i, 0]], data_x[[ind_i, 1]],
            data_x[[ind_i, 2]], class_t[ind_i])
    }

    // run gradeint descent
    let weights_w = Vector::<f64>::zeros(num_dim+1);
    let mut output_y = Vector::<f64>::zeros(num_data_points);
    let max_iter: usize = 10;
    for iter_i in 0..max_iter {
        println!("iteration number {}", iter_i);
        // compute all the activations
        for ind_i in 0..num_data_points {
            let data_x_row = data_x.row_mut(ind_i);
            let data_x_row_1 : Vector<f64> = data_x_row.into();
            let a = data_x_row_1.dot(&weights_w);

            // compute outputs
            output_y[ind_i] = sigmoid(a);
        }

        // compute erros
        // let err = &class_t - &output_y;

        // compute the gradeint vector
        // make step, using learning rate eta and weight decay alpha
    }
}
