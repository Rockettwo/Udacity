#include "PID.h"
#include <cmath>
#include <iostream>

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_, double maxVal_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;

  p_error = 0;
  i_error = 0;
  d_error = 0;

  maxVal = maxVal_;
}

void PID::UpdateError(double cte) {
  d_error = cte - p_error;
  p_error = cte;
  i_error = i_error * 1 + cte;
}

double PID::TotalError() {
  double pid_out = -Kp * p_error - Kd * d_error - Ki * i_error;

  if (pid_out < -maxVal) 
    pid_out = -maxVal;
  else if (pid_out > maxVal)
    pid_out = maxVal; 

  return pid_out;
}

