/* 
Differential Privacy mechanisms
*/
#ifndef XGBOOST_MECHANISMS_H_
#define XGBOOST_MECHANISMS_H_

#include <random>
#include <cmath> 
#include <vector>
#include <mutex>
#include <ctime>

#include "../../src/tree/param.h"

namespace xgboost
{
  using SplitEntry = tree::SplitEntry;

  enum {
    DP_EXPONENTIAL_MECH = 1,
    DP_LAPLACE_MECH = 2,
    DP_HISTOGRAM_MECH = 3
  };

  typedef struct DPMechanism{
    int mech_type;
    double epsilon;
  } DPMechanism; 

  double inline UniformDouble(std::mt19937& gen) {
    // TODO: this should be modified to use Google DP's implementation
    std::uniform_real_distribution<double> dist(0.0,1.0);

    double v = dist(gen);
    return v; 
  }

  size_t inline exponentialMech(std::mt19937& gen, const std::vector<SplitEntry>& candidates, double epsilon = 1.0,
                    double sensitivity = 3.0) {
    // choose index with the racing method proposed by Mironov
    // same as Gumbel trick
    // https://arxiv.org/pdf/2102.08244.pdf 

    const int N = candidates.size();
    size_t index = 0, true_min_index = 0;
    float curr_min = std::numeric_limits<double>::infinity();
    
    for(int i = 0; i<N; i++) {
      double U = UniformDouble(gen); 
      double value = log(log(1.0/U)) - (epsilon * candidates[i].loss_chg)/(2*sensitivity);
      if(value < curr_min) {
        index = i;
        curr_min = value;
      }

    }
    return index;
  }

  /* 
    The following comes from Google DP repo https://github.com/google/differential-privacy
    Laplace noise generation as described in:
    https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf
  */

  const double GRANULARITY_PARAM = static_cast<double>(int64_t{1} << 40);

  double inline GetNextPowerOfTwo(double n) { return pow(2.0, ceil(log2(n))); }

  bool inline boolSample(std::mt19937& gen, double p = 0.5) {
    double q = UniformDouble(gen);
    if(q < p) {
      return true;
    }
    return false;
  }

  int64_t inline geoSample(double lambda, std::mt19937& gen) {
    if (lambda == std::numeric_limits<double>::infinity()) {
      return 0;
    }

    if (UniformDouble(gen) >
        -1.0 * expm1(-1.0 * lambda * std::numeric_limits<int64_t>::max())) {
      return std::numeric_limits<int64_t>::max();
    }

    // Performs a binary search for the sample over the range of possible output
    // values. At each step we split the remaining range in two and pick the left
    // or right side proportional to the probability that the output falls within
    // that range, ending when we have only a single possible sample remaining.
    int64_t lo = 0;
    int64_t hi = std::numeric_limits<int64_t>::max();
    while (hi - lo > 1) {
      int64_t mid =
          lo -
          static_cast<int64_t>(std::floor(
              (std::log(0.5) + std::log1p(exp(lambda * (lo - hi)))) / lambda));
      mid = std::min(std::max(mid, lo + 1), hi - 1);

      double q = std::expm1(lambda * (lo - mid)) / expm1(lambda * (lo - hi));
      if (UniformDouble(gen) <= q) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    return hi - 1;
}

double inline addLaplaceNoise(std::mt19937& gen, const double value, const double epsilon = 1.0,
                        const double sensitivity = 1.0) {
  double gran = GetNextPowerOfTwo((sensitivity / epsilon) / GRANULARITY_PARAM);
  
  double lambda = gran * epsilon / (sensitivity + gran);

  int64_t sample;
  bool sign;
  do {
    sample = geoSample(lambda, gen);
    sign = boolSample(gen);
  } while (sample == 0 && !sign);
  sample = sign ? sample : -sample;

  double noise = sample * gran;
  return value + noise;
}

double inline noisyAverage(std::mt19937& gen, double sum, double count, double epsilon = 1.0, 
  double min = 0.0, double max = 1.0) {
    // Li (2016) book
    // Differential Privacy from theory to practice 
    // Case study: average (algorithm 2.3) here we now count is always > 1 

    double A = addLaplaceNoise(gen, sum, epsilon, max-min) / count;

    if(A > max) A = max; 
    if(A < min) A = min; 

    return A; 
}


}; // namespace xgboost

#endif
