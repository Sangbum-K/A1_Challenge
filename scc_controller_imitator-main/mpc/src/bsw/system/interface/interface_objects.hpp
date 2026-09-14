/**
 * @file        interface_objects.hpp
 * @brief       interface for objects
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 *              2024-04-30 updated by Yuseung Na: Add 2D objects
 */

#ifndef __INTERFACE_OBJECTS_HPP__
#define __INTERFACE_OBJECTS_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>

// Interface Header
#include "interface_std_msgs.hpp"
#include "interface_geometry_msgs.hpp"

namespace interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // enum
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //     
    typedef enum {
        RED_YELLOW_GREEN        = 0,
        RED_YELLOW_LEFT         = 1,
        RED_YELLOW_LEFT_GREEN   = 2,
        YELLOW_YELLOW_YELLOW    = 100
    } TrafficLightType;

    typedef enum {
        RED          = 1,
        YELLOW       = 4,
        YELLOW_53248 = 53248,
        YELLOW_57344 = 57344,
        RED_YELLOW   = 5,
        GREEN        = 16,
        RED_GREEN    = 17,        
        YELLOW_GREEN = 20,
        LEFT         = 32,
        RED_LEFT     = 33,
        YELLOW_LEFT  = 36,
        GREEN_LEFT   = 48,
    } TrafficLightColor;    
    
    typedef enum {
        //  UNKNOWN : 0
        UNKNOWN = 0,

        //  TRAFFIC COMPONENTS : 1~9
        TRAFFIC_LIGHT   = 1,
        TRAFFIC_SIGN    = 2,

        //  DYNAMIC OBJECT : 10~19
        DYNAMIC_UNKNOWN = 10,
        CAR = 11,
        TRUCK = 12,
        BUS = 13,
        MOTORCYCLE = 14,
        PEDESTRIAN = 15,

        //  STATIC OBJECT : 20~29
        STATIC_UNKNOWN = 20,
        STATIC_BOLLARD = 21,
        STATIC_CONE = 22,
        STATIC_BOX = 23,
        STATIC_BARRICADE_CONCRETE = 24,
        STATIC_BARRICADE_PLASTIC = 25,
        STATIC_ROCK = 26,
        STATIC_CONSTRUCTION = 27,
        STATIC_MIXER = 28,
        STATIC_BUS = 29,

        // ETC : 30~
    } ObjectClass;

    typedef enum {
        UNKNOWN_STATE = 0,
        STATIC = 1,
        DYNAMIC = 2,
    } ObjectDynamicState;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    typedef struct {
        Header            header;
        std::string       id;
        TrafficLightType  type;
        TrafficLightColor color;
    } TrafficLight;

    typedef struct {
        Header header;
        float x;
        float y;
    } Object2DState;

    typedef struct {
        Header header;
        float x{0.0};
        float y{0.0};
        float z{0.0};
        float vx{0.0};
        float vy{0.0};
        float vz{0.0};
        float ax{0.0};
        float ay{0.0};
        float az{0.0};
        float roll{0.0};
        float pitch{0.0};
        float yaw{0.0};
        float roll_rate{0.0};
        float pitch_rate{0.0};
        float yaw_rate{0.0};
    } Object3DState;

    typedef struct {
        float length;
        float width;
        float height;
    } ObjectDimension;

    typedef struct {
        uint32_t    id;
        float       confidence_score;
        ObjectClass classification;

        ObjectDimension dimension;  
        Object2DState   state;
    } DetectObject2D;

    typedef struct {        
        Header header;
        std::vector<DetectObject2D> object;
    } DetectObjects2D;

    typedef struct {
        uint32_t    id;
        float       confidence_score;
        ObjectClass classification;

        ObjectDimension dimension;        
        Object3DState   state;
    } DetectObject3D;

    typedef struct {        
        Header header;
        std::vector<DetectObject3D> object;
    } DetectObjects3D;

    typedef struct {
        uint32_t            id;
        ObjectClass         classification;
        ObjectDynamicState  dynamic_state;

        ObjectDimension dimension;
        Object3DState   state;
        Object3DState   state_covariance;
    } TrackObject;

    typedef struct {
        Header header;
        std::vector<TrackObject> object;
    } TrackObjects;

    typedef struct {
        std::vector<Point> point;
    } ReferenceTrajectory;

    typedef struct {
        double                      probability;
        std::vector<Object3DState>  state;
    } PredictObjectMultimodal;

    typedef struct {
        uint32_t            id;
        ObjectClass         classification;
        ObjectDynamicState  dynamic_state;
        std::vector<ReferenceTrajectory> reference;
        
        ObjectDimension         dimension;

        // Multi-modal Prediction
        std::vector<PredictObjectMultimodal> state_multi;

        // Highest Probability
        std::vector<Object3DState>      state;
    } PredictObject;

    typedef struct {
        Header header;
        std::vector<PredictObject> object;
    } PredictObjects;

    typedef struct {
        Header header;
        bool is_valid;
        double lx;
        double ly;
        double rx;
        double ry;
        double dist;
        double remain_dist;
    } TargetLine;

    typedef struct{
        Header header;
        std::vector<TargetLine> line;
    } TargetLines;

    typedef struct {
        Header header;
        
        uint32_t id;
        ObjectDimension dimension;
        ObjectDynamicState dynamic_state;

        double dist{0.0};
        double buffer{0.0};

        double s{0.0};
        double closest_s{0.0};
        double ds{0.0};

        double n{0.0};
        double closest_n{0.0};
        double dn{0.0};
    } TargetFrenetObject;
} // namespace interface

#endif // __INTERFACE_OBJECTS_HPP__
