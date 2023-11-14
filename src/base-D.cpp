#include <chrono>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "../include/rapidcsv.h"

const int nParticles = 1<<nParticles_pow;
const int D = 2; 

class doubD {
private:
    double* start;
    int D;
public:
    doubD() : start(nullptr), D(0) {}

    doubD(int D, double val) : start(new double[D]), D(D) {
        for (int i = 0; i < D; ++i) {
            start[i] = val;
        }
    }

    doubD(int D, double vals[]) : start(new double[D]), D(D) {
        for (int i = 0; i < D; ++i) {
            start[i] = vals[i];
        }
    }

    doubD(const doubD& other) : start(new double[other.D]), D(other.D) {
        for (int i = 0; i < D; ++i) {
            start[i] = other.start[i];
        }
    }

    doubD& operator=(const doubD& other) {
        if (this != &other) {
            delete[] start;
            start = new double[other.D];
            D = other.D;
            for (int i = 0; i < D; ++i) {
                start[i] = other.start[i];
            }
        }
        return *this;
    }

    double& operator[](int idx) {
        return start[idx];
    }

    const double& operator[](int idx) const {
        return start[idx];
    }

    ~doubD() {
        delete[] start;
    }


    friend std::ostream& operator<<(std::ostream& os, const doubD& vec) {
        os << "(";
        for (int i = 0; i < vec.D - 1; ++i) {
            os << vec.start[i] << ", ";
        }
        os << vec.start[vec.D - 1] << ")";
        return os;
    }
};


struct swarm {
    doubD position[nParticles];
    doubD velocity[nParticles];
    doubD localBest[nParticles];
}sw;

double f(doubD position){
return -10 * (position[0]/5 - pow(position[0], 3) - pow(position[1], 5) ) * exp(-1 * pow(position[0], 2) - pow(position[1], 2) );
}

int main(int argc, char *argv[]) {

    const int NITER = 10000;
    const double a = 0.72984;
    const double bGlob = 1.49617;
    const double bLoc = 1.49617;
    const int c = 1;
    const int d = 1;

    const doubD maxDims{D, 100.0};
    doubD globalBest{D, 0.0};
    double valueGlobalBest = 1000.0;

    rapidcsv::Document doc("include/data_X.txt", rapidcsv::LabelParams(-1, -1));

    long long volume = doc.GetCell<long long>(4, 2);
    for (int i = 0; i < nParticles; ++i) {
        double psoition_vals[] = {1.0 * (rand() % int(maxDims[0]))-(maxDims[0]/2), 1.0 *( (rand() % int(maxDims[1]))-(maxDims[1]/2))};
        sw.position[i] = doubD(D, psoition_vals);
        sw.velocity[i] = doubD(D, 1);
        sw.localBest[i] = sw.position[i];
    }



    std::srand(42);
    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < NITER; ++iter) {
        double rGlob = ((double)rand()) / RAND_MAX;
        double rLoc = ((double)rand()) / RAND_MAX;
        for(int p = 0; p < nParticles; ++p) {
            double velocity_vals[] = {a*sw.velocity[p][0] + bGlob*rGlob*(globalBest[0] - sw.position[p][0]) + bLoc*rLoc*(sw.localBest[p][0] - sw.position[p][0]),
                                    a*sw.velocity[p][1] + bGlob*rGlob*(globalBest[1] - sw.position[p][1]) + bLoc*rLoc*(sw.localBest[p][1] - sw.position[p][1])};
            sw.velocity[p] = doubD(D, velocity_vals);

            double position_vals[] = {c*sw.position[p][0] + d*sw.velocity[p][0], 
                                    c*sw.position[p][1] + d*sw.velocity[p][1]};

            sw.position[p] = doubD(D, position_vals);

        }

        for(int p = 0; p < nParticles; ++p) {
            if(f(sw.position[p]) <= f(sw.localBest[p])){
                sw.localBest[p] = sw.position[p];
            }
            if(f(sw.position[p]) <= valueGlobalBest){
                globalBest = sw.position[p];
                valueGlobalBest = f(sw.position[p]);
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout <<nParticles<<"\t"<< (end-start).count()/NITER/1000.0  << std::endl;

    // std::cout << globalBest << '\n';

    #ifdef DEBUG
    std::cout << "The minimum value is --> " << valueGlobalBest << " and the coordinate is " << globalBest[0] << " " << globalBest[1] << std::endl;
    #endif

    return 0;
}
