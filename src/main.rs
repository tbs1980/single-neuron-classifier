extern crate rulinalg;
extern crate rand;

// use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::BaseMatrixMut;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use rulinalg::norm::Euclidean;
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
                class_t[ind_i] = 0.0_f64
            } else {
                data_x[[ind_i,ind_j]] = v + 3.0_f64;
                class_t[ind_i] = 1.0_f64
            }
        }
        // println!("{}\t{}\t{}", data_x[[ind_i, 1]], data_x[[ind_i, 2]], class_t[ind_i])
    }

    // run gradeint descent
    let mut weights_w = Vector::<f64>::zeros(num_dim+1);
    let mut grad_weights_w = Vector::<f64>::zeros(num_dim+1);
    let mut output_y = Vector::<f64>::zeros(num_data_points);
    let max_iter: usize = 1000;
    let learning_rate_eta = 1e-3;
    let regularisation_alpha = 1e-3;
    for iter_i in 0..max_iter {
        // compute all the activations
        for ind_i in 0..num_data_points {
            let data_x_row = data_x.row_mut(ind_i);
            let data_x_row_1 : Vector<f64> = data_x_row.into();
            let a = data_x_row_1.dot(&weights_w);

            // compute outputs
            output_y[ind_i] = sigmoid(a);
        }

        // compute erros
        let err = &class_t - &output_y;

        // compute the gradeint vector
        for ind_j in 0..num_dim+1 {
            let mut grad_wt = 0.0_f64;
            for ind_i in 0..num_data_points {
                grad_wt -= err[ind_i]*data_x[[ind_i, ind_j]];
            }
            grad_weights_w[ind_j] = grad_wt;
        }

        // make step, using learning rate eta and weight decay alpha
        weights_w = &weights_w - ( &grad_weights_w + &weights_w*regularisation_alpha )*learning_rate_eta;

        println!("iteration {}, norm  = {}", iter_i , grad_weights_w.norm(Euclidean));
    }

    println!("final weights are {}", weights_w);
}
