#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

namespace {
struct Cone {
    Point2f center;
    float area;
};

const Scalar BLUE_LOWER(95, 110, 70);
const Scalar BLUE_UPPER(130, 255, 245);

const Scalar YELLOW_LOWER(20, 70, 140);
const Scalar YELLOW_UPPER(34, 255, 255);

const int MIN_CONE_AREA = 120;
const int MIN_CONE_WIDTH = 6;
const int MAX_CONE_WIDTH = 80;
const int MIN_CONE_HEIGHT = 12;
const int MAX_CONE_HEIGHT = 90;

const float MAX_PAIR_VERTICAL_GAP = 30.0f;
const int SAMPLE_STRIDE = 5;
const float GAUSSIAN_SIGMA = 18.0f;

Rect buildGroundRoi(const Size &size) {
    int roiY = static_cast<int>(size.height * 0.45f);
    int roiH = size.height - roiY;
    return Rect(0, roiY, size.width, roiH);
}

Mat preprocessMask(const Mat &inputMask) {
    Mat mask = inputMask.clone();
    medianBlur(mask, mask, 5);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    return mask;
}

vector<Cone> detectCones(const Mat &frame, const Scalar &lower, const Scalar &upper) {
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    Rect roi = buildGroundRoi(frame.size());
    Mat hsvRoi = hsv(roi);

    Mat mask;
    inRange(hsvRoi, lower, upper, mask);
    mask = preprocessMask(mask);

    Mat sharpMask;
    GaussianBlur(mask, sharpMask, Size(0, 0), 1.2);
    addWeighted(mask, 1.3, sharpMask, -0.3, 0, mask);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Cone> cones;
    for (const auto &contour : contours) {
        float area = static_cast<float>(contourArea(contour));
        if (area < MIN_CONE_AREA) {
            continue;
        }
        Rect bounds = boundingRect(contour);
        if (bounds.width < MIN_CONE_WIDTH || bounds.width > MAX_CONE_WIDTH) {
            continue;
        }
        if (bounds.height < MIN_CONE_HEIGHT || bounds.height > MAX_CONE_HEIGHT) {
            continue;
        }
        Moments mu = moments(contour);
        if (mu.m00 == 0) {
            continue;
        }
        Point2f center(static_cast<float>(mu.m10 / mu.m00) + roi.x,
                       static_cast<float>(mu.m01 / mu.m00) + roi.y);

        if (center.y < roi.y + 10) {
            continue;
        }
        cones.push_back({center, area});
    }
    return cones;
}

vector<Point2f> pairCones(const vector<Cone> &blueCones, const vector<Cone> &yellowCones) {
    vector<Point2f> midPoints;
    vector<int> yellowUsed(yellowCones.size(), 0);

    vector<Cone> sortedBlue = blueCones;
    sort(sortedBlue.begin(), sortedBlue.end(),
         [](const Cone &a, const Cone &b) { return a.center.y < b.center.y; });

    for (const auto &blue : sortedBlue) {
        float bestGap = MAX_PAIR_VERTICAL_GAP;
        int bestIdx = -1;
        for (size_t i = 0; i < yellowCones.size(); ++i) {
            if (yellowUsed[i]) {
                continue;
            }
            float gap = static_cast<float>(fabs(blue.center.y - yellowCones[i].center.y));
            if (gap < bestGap) {
                bestGap = gap;
                bestIdx = static_cast<int>(i);
            }
        }
        if (bestIdx >= 0) {
            yellowUsed[bestIdx] = 1;
            Point2f midpoint((blue.center.x + yellowCones[bestIdx].center.x) * 0.5f,
                             (blue.center.y + yellowCones[bestIdx].center.y) * 0.5f);
            midPoints.push_back(midpoint);
        }
    }

    sort(midPoints.begin(), midPoints.end(),
         [](const Point2f &a, const Point2f &b) { return a.y > b.y; });
    return midPoints;
}

void drawCones(Mat &canvas, const vector<Cone> &cones, const Scalar &color, const string &label) {
    for (const auto &cone : cones) {
        circle(canvas, cone.center, 6, color, 2, LINE_AA);
    }
    putText(canvas, label + ": " + to_string(cones.size()),
            Point(10, 20 + (label == "Blue" ? 0 : 20)),
            FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
}

void drawPath(Mat &canvas, const vector<Point2f> &midPoints) {
    if (midPoints.size() < 2) {
        return;
    }
    for (size_t i = 1; i < midPoints.size(); ++i) {
        line(canvas, midPoints[i - 1], midPoints[i], Scalar(0, 255, 0), 2, LINE_AA);
    }
    for (const auto &pt : midPoints) {
        circle(canvas, pt, 4, Scalar(0, 200, 0), FILLED, LINE_AA);
    }
}

vector<Point2f> interpolatePath(const vector<Point2f> &midPoints, const Size &size) {
    if (midPoints.empty()) {
        return {};
    }
    vector<Point2f> sortedMid = midPoints;
    sort(sortedMid.begin(), sortedMid.end(),
         [](const Point2f &a, const Point2f &b) { return a.y > b.y; });

    vector<Point2f> spline;
    for (int y = static_cast<int>(sortedMid.back().y);
         y >= 0; y -= SAMPLE_STRIDE) {
        float accumX = 0.0f;
        float weightSum = 0.0f;
        for (const auto &pt : sortedMid) {
            float dist = fabs(pt.y - y);
            float weight = std::exp(-dist * dist / (2.0f * GAUSSIAN_SIGMA * GAUSSIAN_SIGMA));
            accumX += weight * pt.x;
            weightSum += weight;
        }
        if (weightSum > 1e-3f) {
            float x = accumX / weightSum;
            spline.emplace_back(Point2f(x, static_cast<float>(y)));
        }
    }
    return spline;
}

void drawSpline(Mat &canvas, const vector<Point2f> &spline) {
    if (spline.size() < 2) {
        return;
    }
    for (size_t i = 1; i < spline.size(); ++i) {
        line(canvas, spline[i - 1], spline[i], Scalar(0, 255, 0), 2, LINE_AA);
    }
}
} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "用法: ./test_cone_path <图片路径>" << endl;
        return -1;
    }

    string imagePath = argv[1];
    Mat input = imread(imagePath, IMREAD_COLOR);
    if (input.empty()) {
        cout << "错误: 无法读取图片 " << imagePath << endl;
        return -1;
    }

    cout << "原始尺寸: " << input.cols << "x" << input.rows << endl;

    Mat frame;
    resize(input, frame, Size(640, 480));
    imshow("1. Resized Frame", frame);
    waitKey(0);

    vector<Cone> blueCones = detectCones(frame, BLUE_LOWER, BLUE_UPPER);
    vector<Cone> yellowCones = detectCones(frame, YELLOW_LOWER, YELLOW_UPPER);

    Mat detectionVis = frame.clone();
    drawCones(detectionVis, blueCones, Scalar(255, 0, 0), "Blue");
    drawCones(detectionVis, yellowCones, Scalar(0, 255, 255), "Yellow");
    imshow("2. Cone Detection", detectionVis);
    waitKey(0);

    vector<Point2f> midPoints = pairCones(blueCones, yellowCones);
    Mat pathVis = frame.clone();
    drawCones(pathVis, blueCones, Scalar(255, 0, 0), "Blue");
    drawCones(pathVis, yellowCones, Scalar(0, 255, 255), "Yellow");
    drawPath(pathVis, midPoints);
    imshow("3. Raw Mid-Points Path", pathVis);
    waitKey(0);

    destroyAllWindows();
    cout << "测试完成!" << endl;
    return 0;
}

