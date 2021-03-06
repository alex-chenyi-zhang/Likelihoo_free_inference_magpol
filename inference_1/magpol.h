#ifndef MAGPOL_H
#define MAGPOL_H

#include <random>
#include <iostream>

namespace magpol{
class linear_hash{
    public:
        int M = 5000000;
        int a[3] = {17,290,4914};
        int * occupancy;
        int * monomer_index;
        int ** key_values;
        linear_hash();
        ~linear_hash();
};

class saw_MC{
    bool compute_summary_stats;
    std::string conf_filename;
    std::string spins_filename;
    std::string features_filename;
    int stride, n_swaps;                           // in the Multiple Markov Chain stride is the number of MC moves before attempting a swap
                                                   // n_steps is set equal to n_swaps*stride 
    int n_mono, n_steps;
    int n_samples;
    int samples_lag;
    int ** coord;                                  // 2D array that contains the coordinates of the monomers
    int ** trial_coord;
    int * Ree2;                                    // Here we store the squared end-to-end distance
    float *Rg2;
    int * spins;                                   // Here I store the spin congiguration of the Ising/Potts subsystem
    int * trial_spins;
    float * h_fields;                              // Values of the local fields in the Ising/Potts model
    float * CpG_fraction;
    float max_field_value = 2;
    float spin_coupling;
    float energy = 0;
    float alpha_h;                           // alpha is between 0 and 1. it is the importance of the polymer part of the hamiltonian vs the 
                                                   // the part that ONLY depends on the spins i.e. the parts with the external fields
    float beta_temp;
    float * energies;
    float * magnetization;
    int p_moves[47][2][3] = {};                // The 47 orthogonal transformations of the octahedral group (except identity)

    int ** neighbours;                             /* This is the list of the neighbours of each monomer. Needed to evaluate the hamiltonian.          
                                                      To compute energy is order n_mono, so is the same order of the other operations needed
                                                      to propose a move, so I guess it's ok. To increase the efficiency one should compute only
                                                      the delta energy, but it's not so straighforward as the pivot is a global move. In this 
                                                      probably a trial_neighbour matrix will be needed*/
    int ** trial_neighbours;

    float ** summary_stats;
    std::random_device rd;                         // generates a random seed, to initialize a random engine
    std::mt19937 mt;
    std::uniform_real_distribution<double> uni_R;  // uniform in [0,1]. Used to acc/rej MC move
    std::uniform_int_distribution<int> uni_I_poly; // uniform in [1,n_mono-1]. To pick a random monomer. We don't do pivots around the 0-th monomer 
    std::uniform_int_distribution<int> uni_I_poly2;// uniform in [1,n_mono-3]. Needed for the one-bead-flip
    std::uniform_int_distribution<int> uni_I_poly3;// uniform in [1,n_mono-4]. Needed for cranckshaft-type moves
    std::uniform_int_distribution<int> uni_G;      // uniform in [0,47-1]. To pick a random element of the symmetry group
    std::uniform_int_distribution<int> uni_spins;  // random number picked between {-1,0,+1}
    std::uniform_int_distribution<int> uni_spins2; // random number picked between {0,1}
    std::uniform_int_distribution<int> local_move_rand;  // allows you to choose one of the 4 implemented local moves
    int perms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    int tr_signs[8][3] = {{1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},       
    {-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}};      // by combining perm's and sign comb's you obtain the 48 pivot moves
    linear_hash hash_saw;                          // We declare here the hash table that will be used for self-avoidance checks
    int * hashed_where;                            // at each attempted pivot you store here the hashed coordinates of the monomers
                                                   // :this is useful for quick cleanup of the hashtable.
    int * whos_hashed;                             // constains the sequence of monomers used for a self-av check
    int n_hashes;                                  // Tells you in each self_av. check how many monomers you inserted in the hash table
    public:
        int done_strides = 0;                          // incremented every time a stride is done. Needed to properly save the observables we need
        int i_sample = 0;
        saw_MC(int,int,int,int,float,float,float,std::string,std::string,std::string,bool);                           //constructor
        ~saw_MC();                                 //destructor
        void try_pivot(int,int);
        int hash_function(int*,int,int*);
        bool check_saw(int); 
        void compute_neighbour_list(int**, int**);          
        float compute_new_energy(int**);
        void spins_MC(void);
        void remove_from_hash_table(int);
        bool add_to_hash_table(int, int*);
        bool check_site_occupancy(int, int*);
        void single_bead_flip(void);
        void crankshaft_180(void);
        void crankshaft_90_270(int);
        void run(void);     
        void write_results_on_file(void);   
        float gyr_rad_square(void);  
        float lag_p_spin_autocovariance(int);  
        float get_beta(void);
        float get_ene(void);
        int get_nsamples(void);
        void copy_spins(int*);
        void copy_coords(int**);   
        void copy_summary_statistics(float**);  
        void set_spins(int*);
        void set_coords(int**);
        void initialize_configuration(bool); // if the argument is false just initialize from straight rod and random spins, otherwise from input files
        void initialize_parameters(float);
        void write_results_on_file2(void); 
};


// This class performs a simulation with the multiple markov chains method and uses as the 
// algorithm for the single chain the code used in the class saw_MC written above
class multiple_markov_chains{
    bool from_files;
    bool initialize_weight;              // If true read the weight value from the input file, otherwise generate randomly
    float regression_weight = 1.0;
    int n_mono, n_samples, stride_length, n_temps, samples_lag, n_strides;
    float J, alpha, beta_temp;
    float min_inv_temp = 0.2;
    std::string conf_filename;
    std::string spins_filename;
    std::string features_filename;
    int * temporary_spins1;
    int * temporary_spins2;
    int ** temporary_coords1;
    int ** temporary_coords2;
    float ** summary_statistics;
    std::random_device rand_dev;                         // generates a random seed, to initialize a random engine
    std::mt19937 mer_twist;
    std::uniform_real_distribution<double> uni_01;  // uniform in [0,1]. Used to acc/rej MC move
    std::uniform_int_distribution<int> uni_temps;
    std::normal_distribution<double> weight_distribution;
    saw_MC** simulations;
    /*std::random_device rand_dev;                         // generates a random seed, to initialize a random engine
    std::mt19937 mer_twist;
    std::uniform_real_distribution<double> uni_01;  // uniform in [0,1]. Used to acc/rej MC move
    std::uniform_int_distribution<int> uni_temps; */  // picks a random pair of neighbouring chains 
    public:
        multiple_markov_chains(std::string);
        ~multiple_markov_chains(void);
        void copy_summary_statistics(double**);  
        void run_MMC(void);
        int get_nsamples(void);
        double get_weight(void);
        void set_weight(double);
};



}
#endif // MAGPOL_H