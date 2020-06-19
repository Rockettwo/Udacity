#include <uWS/uWS.h>
#include <fstream>
#include "json.hpp"
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "spline.h"
#include <algorithm>


// for convenience
using nlohmann::json;
using std::string;
using std::vector;


typedef struct Trajectory { 
	vector<double> x_pts;
	vector<double> y_pts;
	double cost;
} Trajectory;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;
	double maxVelo = 49.5;
	double velocity = 0;
	double collFreeCount = 0;
	
	bool following = false;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&collFreeCount,&maxVelo,&velocity,&following,&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];
					
					int prev_size = previous_path_x.size();
 
          json msgJson;
          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */

					// points to take over from prev path
					vector<double> ptsx;
					vector<double> ptsy;
					
					int numOfPreds = 80;		// number of path points in output traj
					int useFirstX = 5;			// number of path points taken over from prev path
					double basicProjectionLength = 30; 	// horizon used for path prediction

					// actual car pose
					double ref_x = car_x;
					double ref_y = car_y;
					double ref_yaw = deg2rad(car_yaw);

					if (prev_size < 2) {
						// not enough points, generate start points tangential to car's dir
						double prev_car_x = car_x - cos(car_yaw);
						double prev_car_y = car_y - sin(car_yaw);

						ptsx.push_back(prev_car_x);
						ptsx.push_back(car_x);

						ptsy.push_back(prev_car_y);
						ptsy.push_back(car_y);
						useFirstX = prev_size;
					} else {
						// use at most 'useFirstX' points from prev traj
						useFirstX = std::min(useFirstX,prev_size);
						ref_x = previous_path_x[useFirstX - 1];
						ref_y = previous_path_y[useFirstX - 1];
						
						double ref_x_prev = previous_path_x[useFirstX - 2];
						double ref_y_prev = previous_path_y[useFirstX - 2];
						ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);
						
						ptsx.push_back(ref_x_prev);
						ptsy.push_back(ref_y_prev);
						ptsx.push_back(ref_x);
						ptsy.push_back(ref_y);
					}
					
					
					// all trajs and special pointer for traj to follow car
					vector<Trajectory*> trajs;
					Trajectory* followTraj;
					
					// increase velocity gradually to not hit the limits
					if (!following && velocity < maxVelo) {
						velocity += 0.224;
					}
					
					// create 4 trajectories for targets (left, middle, right) and low speed follow
					for (int i = 0; i < 4; ++i) {
						Trajectory* traj = new Trajectory();
						vector<double> ptsx_t; ptsx_t = ptsx;
						vector<double> ptsy_t; ptsy_t = ptsy;
						
						// get current lane of vehicle for follow Traj
						int lane = i;						
						vector<double> start_f =  getFrenet(car_x, car_y, ref_yaw, map_waypoints_x, map_waypoints_y);
						if (i == 3) {
							lane = int (start_f[1] / 4);
							followTraj = traj;
						} 
						
						// create four additional points to fit spline
						for (int h = 1; h < 4; ++h) {
							vector<double> next_wp = getXY(car_s + h * basicProjectionLength, (2 + lane * 4), map_waypoints_s, map_waypoints_x, map_waypoints_y);								
							ptsx_t.push_back(next_wp[0]); ptsy_t.push_back(next_wp[1]);
						}

						// shift car reference angle to 0 degree
						for (int h = 0; h < ptsx_t.size(); ++h) {
							double shift_x = ptsx_t[h] - ref_x;
							double shift_y = ptsy_t[h] - ref_y;

							ptsx_t[h] = (shift_x * cos(ref_yaw) - shift_y * sin(-ref_yaw));
							ptsy_t[h] = (shift_x * sin(-ref_yaw) + shift_y * cos(ref_yaw));
						}

						// create a spline and set points
						tk::spline s;
						s.set_points(ptsx_t, ptsy_t);

						// start with the previous path points from last time
						for (int h = 0; h < useFirstX; ++h) {
							traj->x_pts.push_back(previous_path_x[h]);
							traj->y_pts.push_back(previous_path_y[h]);
						}
										
						// special case: if following decrease velocity to 
						if (i == 3 && following) {
							double minV = 100000;
							for (auto v : sensor_fusion) {
								double vx = v[3]; double vy = v[4];
								float s = v[5]; float d = v[6];
								double v_tot = std::sqrt((vx * vx) + (vy * vy));
								int tmp_l = int (d / 4);
								if (lane == tmp_l && distance(s,d,start_f[0],start_f[1]) < 100 && v_tot < velocity) 
									minV = v_tot;
							}
							if (velocity > minV - 2 && !(velocity < maxVelo/3))
								velocity -= 0.224;
						}						
						
						// calculate target
						double target_y = s(basicProjectionLength);
						double target_dist = sqrt((basicProjectionLength) * (basicProjectionLength) + (target_y) * (target_y));
						double x_cum = 0;
						
						// The rest after filling it with previous points
						for (int h = 0; h < numOfPreds - useFirstX; ++h) {
							double N = (target_dist / (0.02 * velocity / 2.24)); 
							double x_point = x_cum + (basicProjectionLength) / N;
							double y_point = s(x_point);
							x_cum = x_point;

							double x_ref = x_point;
							double y_ref = y_point;

							// rotating back to normal after rotating it earlier
							x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
							y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

							x_point += ref_x;
							y_point += ref_y;

							traj->x_pts.push_back(x_point);
							traj->y_pts.push_back(y_point);
						}
												
						// calculate cost
						for (int h = 1; h <= numOfPreds; ++h) {
							// predict collision with all vehicles
							for (auto v : sensor_fusion) {
								double vx = v[3]; double vy = v[4];
								float s = v[5]; float d = v[6];
								
								double ds = std::sqrt((vx * vx) + (vy * vy)) * h * 0.02;
								
								vector<double> xy_f =  getFrenet(traj->x_pts[h-1], traj->y_pts[h-1], ref_yaw, map_waypoints_x, map_waypoints_y);

								// high cost if collision is detected
								if (distance(0, d, 0, xy_f[1]) < 2 && distance(s+ds, d, xy_f[0], xy_f[1]) < 10) {
									traj->cost += 11000;
								}
							}
						}
						
						lane = int (start_f[1] / 4);						
						if (!following && lane != i) {
							traj->cost += 1;
						}
						
						if (following && i != 3)
							traj->cost += 500;
						else if (!following && i == 3)
							traj->cost += 100000;
						
						trajs.push_back(traj);
					}
					
					std::sort(trajs.begin(), trajs.end(), [] (Trajectory* a, Trajectory* b) { return a->cost < b->cost; });
					
					if(trajs[0]->cost >= 25000) {
						if (collFreeCount < 10) {
							following = true;
						} else {
							collFreeCount = 0;
						}
					} else {
						if (collFreeCount > 10)  {
							following = false;
						} else {
							++collFreeCount;
						}
							
					}
					
					if (following) {						
						msgJson["next_x"] = followTraj->x_pts;
						msgJson["next_y"] = followTraj->y_pts;
					} else {
						msgJson["next_x"] = trajs[0]->x_pts;
						msgJson["next_y"] = trajs[0]->y_pts;						
					}

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}