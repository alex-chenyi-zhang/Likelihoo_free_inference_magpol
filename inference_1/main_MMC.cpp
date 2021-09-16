#include "magpol.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <fstream>

using namespace std::chrono;
using namespace magpol;

void compute_ss_mean_estimates(int n, double mean[], double **ss_samples){
    for (int j = 0; j < 3; j++){ mean[j] = 0;}

    for (int j = 0; j < 3; j++){
        for (int i = 0; i < n; i++){
            mean[j] += ss_samples[i][j];
        }
        mean[j] = mean[j]/n;
    }
}

void compute_covariance_matrix(int n, double mean[], double **ss_samples, double cov_mat[][3]){
    for (int j = 0; j < 3; j ++){
        for (int l = 0; l < 3; l ++){
            cov_mat[j][l] = 0;
            for (int i = 0; i < n; i++){
                cov_mat[j][l] += (ss_samples[i][j]-mean[j])*(ss_samples[i][l]-mean[l]);
            }
            cov_mat[j][l] = cov_mat[j][l]/(n-1);
        }
    }
}

double compute_inverse_matrix(double m[][3], double invm[][3]){
    double adjugate[3][3];
    adjugate[0][0] =  m[1][1]*m[2][2] - m[2][1]*m[1][2];
    adjugate[1][0] = -m[1][0]*m[2][2] + m[2][0]*m[1][2];
    adjugate[2][0] =  m[1][0]*m[2][1] - m[2][0]*m[1][1];
    adjugate[0][1] = -m[0][1]*m[2][2] + m[2][1]*m[0][2];
    adjugate[1][1] =  m[0][0]*m[2][2] - m[2][0]*m[0][2];
    adjugate[2][1] = -m[0][0]*m[2][1] + m[2][0]*m[0][1];
    adjugate[0][2] =  m[0][1]*m[1][2] - m[1][1]*m[0][2];
    adjugate[1][2] = -m[0][0]*m[1][2] + m[1][0]*m[0][2];
    adjugate[2][2] =  m[0][0]*m[1][1] - m[1][0]*m[0][1]; 
    double det = m[0][0]*adjugate[0][0] + m[0][1]*adjugate[1][0] + m[0][2]*adjugate[2][0];
    if (det == 0){std::cout << "WARNING: TRYING TO INVERSE SINGULAR MATRIX!!!!\n\n\n";}

    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            invm[i][j] = adjugate[i][j]/det;
        }
    }
    std::cout << "DET   " << -det << "\n";
    if (det <= 0){ det = -det;}
    //std::cout << "LOGDET   " << -0.5*log(det) << "\n\n";
    return det;
}

double log_synth_likelihood(double invm[][3], double mean[], double data[], double abs_det){
    double temp[3];
    for (int i = 0; i < 3; i++){
        temp[i] = 0;
        for (int j = 0; j < 3; j++){
            temp[i] += invm[i][j]*(data[j]-mean[j]);
        }
    }
    double log_sl = 0;

    for (int j = 0; j < 3; j++){
        log_sl += (data[j]-mean[j]) * temp[j];
    }
    log_sl = -0.5 * (log_sl + log(abs_det));
    return log_sl;
}




/**************************************************************************************************************************/
int main(int argc, char* argv[]){
    std::stringstream string1(argv[1]);
    std::string input_filename = "input_file.txt";
    string1 >> input_filename;

    std::random_device rand_dev;                         // generates a random seed, to initialize a random engine
    std::mt19937 mer_twist(rand_dev());
    std::uniform_real_distribution<double> uni_01;
    double delta_w = 0.3; // to pick trial weight in the MCMC choose w_trial = w_old + delta_weight*(uni_01(mer_twist)*2-1)
    int n_param_samples = 2000; //number of samples in the parameter space. sampling done with likelihood free techniques
    double ** ss_mean_estimates = new double*[n_param_samples];    // here there are the mean estimates of the summary statistics for each MCMC step in parameter space 
    double * ss_abs_covariance_determinant = new double[n_param_samples];  // here the same thing as above but it's the module of the determinant of the empirical covariance matrix
    double * weight_series = new double[n_param_samples];
    double * syn_likelihoods = new double[n_param_samples];
    for (int i = 0; i < n_param_samples; i++){
        ss_mean_estimates[i] = new double[3];
    }
    double covariance_matrix[3][3]; 
    double inverse_covariance_matrix[3][3];

    multiple_markov_chains MMC_simulation(input_filename);
    int number_samples = MMC_simulation.get_nsamples();
    double ** summary_statistics = new double*[number_samples];
    for (int i = 0; i < number_samples; i++){
        summary_statistics[i] = new double[3];
    }

    /*****************************************************/
    // Here I get the summary statistics from a sample configuration coming out of a simulation with a fixed value of the weight
    // We want to see if we're able to estimate this "ground truth value" by sampling from the approximate synthetic likelihood
    double ss_data[3];
    std::ifstream config_file;
    config_file.open("ss_data.txt");
    if (config_file.is_open()){
        std::string line;
        int i_ss = 0;
        while(std::getline(config_file, line)){
            std::stringstream ss(line);
            if (!(ss >> ss_data[i_ss])){ break; }
            i_ss++;
        }
    }
    else{
        std::cout << "Unable to open the fields file \n";
    }
    config_file.close(); 

    for (int i = 0; i < 3; i++){
        std::cout << ss_data[i] << "\n";
    }
    /*****************************************************/

  
    auto start = high_resolution_clock::now();
    
    double syn_like;
    double weight;
    // perform the first run to get a initial value of the likelihood
    MMC_simulation.run_MMC();
    MMC_simulation.copy_summary_statistics(summary_statistics);
    weight = MMC_simulation.get_weight();
    compute_ss_mean_estimates(number_samples, ss_mean_estimates[0], summary_statistics);
    compute_covariance_matrix(number_samples, ss_mean_estimates[0], summary_statistics, covariance_matrix);
    ss_abs_covariance_determinant[0] = compute_inverse_matrix(covariance_matrix, inverse_covariance_matrix);  // this return abs(det) and at the same time computes the inverse        
    syn_like = log_synth_likelihood(inverse_covariance_matrix, ss_mean_estimates[0], ss_data, ss_abs_covariance_determinant[0]); 
    
    syn_likelihoods[0] = syn_like;
    weight_series[0] = weight;


    double trial_w;
    double trial_syn_like;
    double delta_syn_like;
    double w_acceptance;
    int accepted_moves = 0;

    // perform sampling in the parameter space where in the metropolis acceptance you use the suynthetic likelihood
    for (int i_param = 1; i_param < n_param_samples; i_param++){
        

        trial_w = weight + (2*uni_01(mer_twist) - 1) * delta_w;
        MMC_simulation.set_weight(trial_w);  // Now one run will be performed witht the trial weight, if rejected the old value needs to be put back

        MMC_simulation.run_MMC();
        MMC_simulation.copy_summary_statistics(summary_statistics);
        compute_ss_mean_estimates(number_samples, ss_mean_estimates[i_param], summary_statistics);
        compute_covariance_matrix(number_samples, ss_mean_estimates[i_param], summary_statistics, covariance_matrix);
        ss_abs_covariance_determinant[i_param] = compute_inverse_matrix(covariance_matrix, inverse_covariance_matrix);  // this return abs(det) and at the same time computes the inverse        
        trial_syn_like = log_synth_likelihood(inverse_covariance_matrix, ss_mean_estimates[i_param], ss_data, ss_abs_covariance_determinant[i_param]); 
        //std::cout << trial_syn_like << "\n";
        //std::cout << ss_abs_covariance_determinant[i_param] << "\n";
        delta_syn_like = trial_syn_like - syn_like;
        //std::cout << delta_syn_like << "\n\n";
        if (delta_syn_like >= 0){ w_acceptance = 1; }
        else { w_acceptance = exp(delta_syn_like); }


        if (i_param%1 == 0){ 
            std::cout << MMC_simulation.get_weight() << "\n";
            std::cout << "Parameter MCMC step: " << i_param << " out of: " << n_param_samples << "\n";
            std::cout << "Current weight: " << weight << "\n";
            std::cout << "Proposed weight: " << trial_w << "\n";
            std::cout << "delta: " << delta_syn_like << "\n";
            std::cout << "current SL: " << syn_like << "\n";
            std::cout << "Proposed SL: " << trial_syn_like << "\n";
            std::cout << "acceptance: " << w_acceptance << "\n\n";
            for (int i = 0; i < 3; i++){ std::cout << ss_mean_estimates[i_param][i] << "\n";}
        }

        if (uni_01(mer_twist) < w_acceptance){
            weight = trial_w;
            syn_like = trial_syn_like;
            accepted_moves++;
        }
        syn_likelihoods[i_param] = syn_like;
        weight_series[i_param] = weight;
    }
    
    std::cout << "Number of accepted moves: " << accepted_moves << " out of " << n_param_samples-1 << "\n";
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << "\n";


    /*******************************************************************/
    std::ofstream myfile;
    myfile.open("weights_series.txt");
    for (int i = 0; i < n_param_samples; i++){ myfile << weight_series[i] << "\n";}
    myfile.close();

    myfile.open("SL_series.txt");
    for (int i = 0; i < n_param_samples; i++){ myfile << syn_likelihoods[i] << "\n";}
    myfile.close();

    myfile.open("cov_determinant_series.txt");
    for (int i = 0; i < n_param_samples; i++){ myfile << ss_abs_covariance_determinant[i] << "\n";}
    myfile.close();

    myfile.open("ss_mean_series.txt");
    for (int i = 0; i < n_param_samples; i++){ 
        for (int j = 0; j < 3; j++){
            myfile << ss_mean_estimates[i][j] << ' ';
        }
        myfile << "\n";
    }
    myfile.close();
    /*******************************************************************/
    for (int i = 0; i < n_param_samples; i++){
        delete [] ss_mean_estimates[i];
    }
    for (int i = 0; i < number_samples; i++){
        delete [] summary_statistics[i];
    }
    delete [] ss_mean_estimates;
    delete [] summary_statistics;
    delete [] ss_abs_covariance_determinant;
    delete [] weight_series;
    delete [] syn_likelihoods;


    return 0;
}