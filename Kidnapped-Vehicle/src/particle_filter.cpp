/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */  
	num_particles = 50;  // TODO: Set the number of particles
	
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_t(theta, std[2]);
	
	
	for (int i = 0; i < num_particles; ++i) {
		Particle p = Particle();
		p.id = i;
		p.x = dist_x(gen); 
		p.y = dist_y(gen);
		p.theta = dist_t(gen);
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	
	//std::cout << "Got to the prediction" << std::endl;
	
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_t(0, std_pos[2]);

	float ydt = yaw_rate * delta_t;		
	float v_ya = velocity / yaw_rate;
	
	for (Particle& p : particles) {
		
		if (fabs(yaw_rate) < 0.0000001) {
			p.x += velocity * delta_t * cos(p.theta) + dist_x(gen);
			p.y += velocity * delta_t * sin(p.theta) + dist_y(gen);
			//p.x += v_ya * (sin(p.theta+ydt) - sin(p.theta)) + dist_x(gen); 
			//p.y += v_ya * (cos(p.theta) - cos(p.theta+ydt)) + dist_y(gen);
			//p.theta += ydt + dist_t(gen);			
			std::cout << "small yaw rate" << std::endl;
		} else {
			p.x += v_ya * (sin(p.theta+ydt) - sin(p.theta)) + dist_x(gen); 
			p.y += v_ya * (cos(p.theta) - cos(p.theta+ydt)) + dist_y(gen);
			p.theta += ydt + dist_t(gen);
		}
	}

}

double ParticleFilter::dataAssociationAndWeight(Particle& p, double std_landmark[],
																		 vector<LandmarkObs> landmarks_fromMap, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
	double w = 1;
	vector<int> associations; 
	vector<double> sense_x; 
	vector<double> sense_y;
	
	for (LandmarkObs& ol : observations) {
		int min_id = -1;
		double min_dist = 100000;
		double map_x = 0.0;
		double map_y = 0.0;
		for (LandmarkObs& gl : landmarks_fromMap) {
			double d = dist(gl.x,gl.y,ol.x,ol.y);
			if (d < min_dist) {
				min_id = gl.id;
				min_dist = d;
				map_x = gl.x; 
				map_y = gl.y;
			}
		}
		if (min_id != -1) {
			w *= multiv_prob(std_landmark[0], std_landmark[1], ol.x, ol.y, map_x, map_y);	
			
			associations.push_back(min_id);
			sense_x.push_back(ol.x);
			sense_y.push_back(ol.y);
		}
	}

	SetAssociations(p,associations,sense_x,sense_y);
	//std::cout << observations.size() << std::endl;
	//std::cout << landmarks_fromMap.size() << std::endl;

	
	// will return 0 if list is empty
	if (associations.size() == 0)
		w = 0;
		
	return w;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
	 
	 double sum_of_weights = 0.0;
	 
		//std::cout << "psize: " << particles.size() << std::endl;
		//std::cout << "wsize: " << particles.size() << std::endl;
		//std::cout << "osize: " << observations.size() << std::endl;
		
		vector<double> w_temp;
		
		
	 for (int i = 0; i < particles.size(); ++i) {
		Particle& p = particles[i];
		
		vector<LandmarkObs> observations_f;
		vector<LandmarkObs> landmarks_fromMap;
		
		// convert to world coordinate
		for (auto ol : observations) {
			LandmarkObs obs;
			obs.x = ol.x * cos(p.theta) - ol.y * sin(p.theta) + p.x;
			obs.y = ol.x * sin(p.theta) + ol.y * cos(p.theta) + p.y;
			observations_f.push_back(obs);
		}
		
		// get only for current range
		for (auto gl : map_landmarks.landmark_list) {
			if (dist(gl.x_f, gl.y_f, p.x, p.y) > sensor_range) 
				continue;
			
			LandmarkObs obs;
			obs.id = gl.id_i;
			obs.x = gl.x_f;
			obs.y = gl.y_f;
			landmarks_fromMap.push_back(obs);
		}
				
		// associate data and weights
		double tmp = dataAssociationAndWeight(p, std_landmark, landmarks_fromMap, observations_f);
		w_temp.push_back(tmp);		
		sum_of_weights += tmp;
	}
	
	if (sum_of_weights < std::numeric_limits<double>::epsilon()) {
		std::cout << "sum_of_weights out of bounds" << std::endl;
		return;
	}
	
	// normalze weights	
	for (int i = 0; i < particles.size(); ++i){
		weights[i] = w_temp[i];	
		weights[i] /= sum_of_weights;
		particles[i].weight = weights[i];
	}

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	std::uniform_real_distribution<double> uniform01(0.0,1.0);
	std::vector<Particle> p_old = particles;
	std::vector<double> w_new = weights;
	
	particles.clear();
	weights.clear();
	
	for (int i = 0; i < p_old.size(); ++i) {
			double rndN = uniform01(gen);
			double tmp = 0.0;
				for (int j = 0; j < w_new.size(); ++j) {
					tmp += w_new[j];
					if (tmp > rndN) {
							particles.push_back(p_old[j]);
							weights.push_back(w_new[j]);
							break;
					}
				}
	}
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}