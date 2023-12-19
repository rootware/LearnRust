use ndarray::prelude::*;
use num_complex::*;
use std::f64::consts::PI;
use ndarray::Zip;


const N_STATES : usize = 11; // Size of matrices
const FREQ : f64 = 20.0; //Sets time for RK in divisions of PI


fn main() {


    let max_state = (N_STATES as f64)-1.0 ; // My states range from max_state to -max_state

    let diagonal = Array::linspace(-max_state, max_state, N_STATES);

    // Is there an easier way to convert the f64 diagonal into a Complex valued array?
    let h0 = (Array2::from_diag(&diagonal)).mapv(|x| Complex64::from(x));

    let mut h1 = Array::from_elem((N_STATES,N_STATES), Complex64::new(0.0,0.0)); //Doing zeros with Complex led to an error
    // Basically, how do I call zeros while also defining the data type to be Complex64

    // POPULATING OFF DIAGONALS
    // In C++, I could do an Eigen operation where I call the relevant h1.block
    // and replace it with an array constructed from linspace and from_diag as above
    for i in 1..(N_STATES -1){
        h1[[i,i+1]] += Complex64::new(-10.0,0.0);
        h1[[i+1,i]] += Complex64::new(-10.0,0.0);
    }

    // define the matrix that governs evolution
    let hamiltonian = h0 + h1 ;


    let mut time: f64 = 0.0; // Keeps track of time
    let period: f64 = PI/FREQ; //period is max value of time for this RK4
    let dt : f64 = period/1e4; // define timestep

    // Define a complex valued vector
    // This is the vector we would like to evolve using RK4
    let mut psi = Array::from_elem(N_STATES, Complex64::new(0.0,0.0));
    psi[5] = Complex64::new(1.0,0.0); //Initialize this vector


    //This function will take in (time, matrix for evolution, current state)
    // and give us the derivative
    fn rhs( t_step: f64, ham: &Array2<Complex<f64>>, wavefunc: &Array1<Complex<f64>>) -> Array1<Complex<f64>>{

        // What I wave is to return -i*t_step*ham*wavefunc, where i is sqrt(-1)
        // However, binary multiplication between complex and ndarrays seems to not be defined
        // hence had to use Zip

        // turn t_step scalar into array
        let f = Array::from_elem(N_STATES,Complex64::new(0.0,-t_step));
        let mut temp = ham.dot(wavefunc);
        // element wise multiplication
        Zip::from(&mut temp )
            .and(&f)
            .for_each(|w, &x| {
                *w *= x ;
            });

        temp
    }


    println!("Initial sum of psi is {}", psi.sum());

    loop {

        let mut newval = psi.clone();
        // I made newval to compute the argument to rhs at every step
        // anyway to optimize this?
        let k1 = rhs(time, &hamiltonian,  &newval);

        // Any way to avoid using Complex64::from every time?
        newval= &psi+&((Complex64::from(dt/2.0))*&k1);
        let k2 = rhs(time + dt/2.0, &hamiltonian, &newval );
        newval = &psi+((Complex64::from(dt/2.0))*&k2);
        let k3 = rhs(time + dt/2.0, &hamiltonian, &newval);
        newval = &psi+((Complex64::from(dt/1.0))*&k3);
        let k4 = rhs(time + dt, &hamiltonian , &newval);

        // RK4 update
        psi = psi + dt/Complex64::from(6.0)*(k1+Complex64::from(2.0)*k2 + Complex64::from(2.0)*k3 + k4);

        time +=dt;
        if time > period{
            break;
        }

    }

    println!("Full wavefunction at end");

    for i in &psi{
        println!("{}",i );
    }

    //Check complex norm, this should be 1
    println!(" Norm at end is: {}",(psi.mapv(|a| a.conj()*a)).sum());
}
