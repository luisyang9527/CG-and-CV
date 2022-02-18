//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v){
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto umin = std::floor(u_img);
        auto umax = std::ceil(u_img);
        auto vmin = std::floor(v_img);
        auto vmax = std::ceil(v_img);

        auto color11 = image_data.at<cv::Vec3b>(vmax, umin);
        auto color12 = image_data.at<cv::Vec3b>(vmax, umax);
        auto color21 = image_data.at<cv::Vec3b>(vmin, umin);
        auto color22 = image_data.at<cv::Vec3b>(vmin, umax);

        auto u_to_umin = (u_img - umin) / (umax - umin);
        auto v_to_vmax = (v_img - vmax) / (vmin - vmax);
        auto uDir1 = (1 - u_to_umin) * color11 + u_to_umin * color12;
        auto uDir2 = (1 - u_to_umin) * color21 + u_to_umin * color22;
        auto color = (1 - v_to_vmax) * uDir1 + v_to_vmax * uDir2;

        return Eigen::Vector3f(color[0], color[1], color[2]);
    }
};
#endif //RASTERIZER_TEXTURE_H
