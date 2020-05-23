#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
							0, 1, 0, 0;
							
	// the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
						
	// the initial covariance matrix P_
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ << 1, 0, 0, 0,
						 0, 1, 0, 0,
						 0, 0, 1000, 0,
						 0, 0, 0, 1000;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			float ro = measurement_pack.raw_measurements_[0];
			float theta = measurement_pack.raw_measurements_[1];
			ekf_.x_ << ro*cos(theta), ro*sin(theta), 0, 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

		previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
		cout << "Initalized" << endl;
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */
	float noise_ax = 9, noise_ay = 9;
	
	// set transition matrix
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
	ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
	
  // set process covariance matrix Q
  MatrixXd G = MatrixXd(4,2);
  G << dt*dt/2, 0, 0, dt*dt/2, dt, 0, 0, dt;
  MatrixXd Qv = MatrixXd(2,2);
  Qv << noise_ax, 0, 0, noise_ay;
  ekf_.Q_ = G * Qv * G.transpose();

  ekf_.Predict();
  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
		cout << "------- radar -------" << endl;
		ekf_.R_ = R_radar_;
		Hj_	= tools.CalculateJacobian(ekf_.x_);
		if (Hj_.isZero(0)) {
			cout << "Jacobian failed" << endl;
			return;
		}
		ekf_.H_ = Hj_;
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);		
  } else {
    // Laser updates
		cout << "------- laser -------" << endl;
		ekf_.R_ = R_laser_;
		ekf_.H_ = H_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);			
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
	previous_timestamp_ = measurement_pack.timestamp_;
}
