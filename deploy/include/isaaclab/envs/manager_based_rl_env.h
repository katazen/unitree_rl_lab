// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include <fstream>
#include "isaaclab/utils/utils.h"

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        action_manager->reset();
        observation_manager->reset();

        debug_csv_.open("debug_log.csv");
        debug_csv_ << "step,type,name,dim";
        for(int i = 0; i < 1000; ++i) debug_csv_ << ",v" << i;
        debug_csv_ << "\n";
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);
        action_manager->process_action(action);

        if(episode_length <= 100) {
            for(const auto& [name, data] : obs) {
                debug_csv_ << episode_length << ",obs," << name << "," << data.size();
                for(const auto& v : data) debug_csv_ << "," << v;
                debug_csv_ << "\n";
            }
            debug_csv_ << episode_length << ",act_raw,," << action.size();
            for(const auto& v : action) debug_csv_ << "," << v;
            debug_csv_ << "\n";
            auto processed = action_manager->processed_actions();
            debug_csv_ << episode_length << ",act_processed,," << processed.size();
            for(const auto& v : processed) debug_csv_ << "," << v;
            debug_csv_ << "\n";
            debug_csv_.flush();
        } else if(debug_csv_.is_open()) {
            debug_csv_.close();
            spdlog::info("Debug log saved to debug_log.csv");
        }
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;

private:
    std::ofstream debug_csv_;
};

};