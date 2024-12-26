#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <iomanip>

static const double PI = 3.141592653589793;

struct Point {
    double x;
    double y;
};

enum class SegmentType { LINE, ARC };
enum class ArcDirection { CW, CCW };

struct Segment {
    SegmentType type;
    Point start;
    Point end;

    // Arc-specific fields
    Point center;  
    double radius; 
    ArcDirection direction; // Clockwise or Counterclockwise
};

// Distance between two points
static double dist(const Point &a, const Point &b) {
    double dx = b.x - a.x, dy = b.y - a.y;
    return std::sqrt(dx*dx + dy*dy);
}

// Signed area * 2 for triangle
static double triangleArea(const Point &p1, const Point &p2, const Point &p3) {
    return 0.5 * ((p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x));
}

// Compute local curvature ~ (4*triangle area)/(product of edges)
static double computeCurvature(const Point &p1, const Point &p2, const Point &p3) {
    double d12 = dist(p1, p2);
    double d23 = dist(p2, p3);
    double d13 = dist(p1, p3);
    double area = std::abs(triangleArea(p1, p2, p3));
    double denom = d12 * d23 * d13;
    if(denom < 1e-12) return 0.0;
    return (4.0 * area) / denom; 
}

// Perpendicular distance from point c to line ab
static double perpendicularDistance(const Point &a, const Point &b, const Point &c) {
    double area = std::abs((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x));
    double base = dist(a, b);
    if(base < 1e-12) return dist(a, c);
    return area / base;
}

// Ramer–Douglas–Peucker simplification
static std::vector<Point> rdpSimplify(const std::vector<Point> &pts, double tolerance) {
    if (pts.size() < 3) return pts;

    double maxDist = 0.0;
    size_t farIndex = 0;
    for (size_t i = 1; i < pts.size() - 1; ++i) {
        double d = perpendicularDistance(pts.front(), pts.back(), pts[i]);
        if (d > maxDist) {
            maxDist = d;
            farIndex = i;
        }
    }

    if (maxDist < tolerance) {
        return { pts.front(), pts.back() };
    } else {
        std::vector<Point> left(pts.begin(), pts.begin() + farIndex + 1);
        std::vector<Point> right(pts.begin() + farIndex, pts.end());
        auto leftSimpl = rdpSimplify(left, tolerance);
        auto rightSimpl = rdpSimplify(right, tolerance);
        leftSimpl.pop_back();
        leftSimpl.insert(leftSimpl.end(), rightSimpl.begin(), rightSimpl.end());
        return leftSimpl;
    }
}

// Discrete Fourier Transform
std::vector<std::complex<double>> dft(const std::vector<double>& data) {
    size_t n = data.size();
    std::vector<std::complex<double>> result(n);
    for(size_t k = 0; k < n; k++) {
        std::complex<double> sum(0,0);
        for(size_t t = 0; t < n; t++) {
            double angle = 2.0 * PI * k * t / double(n);
            sum += std::complex<double>(data[t] * std::cos(angle), -data[t] * std::sin(angle));
        }
        result[k] = sum;
    }
    return result;
}

// Inverse DFT
std::vector<double> idft(const std::vector<std::complex<double>>& freqData) {
    size_t n = freqData.size();
    std::vector<double> result(n, 0.0);
    for(size_t t = 0; t < n; t++) {
        std::complex<double> sum(0,0);
        for(size_t k = 0; k < n; k++) {
            double angle = 2.0 * PI * k * t / double(n);
            sum += std::complex<double>(
               freqData[k].real() * std::cos(angle) - freqData[k].imag() * std::sin(angle),
               freqData[k].real() * std::sin(angle) + freqData[k].imag() * std::cos(angle)
            );
        }
        result[t] = sum.real() / double(n);
    }
    return result;
}

// Zero out high frequency components
void applyFFTFilter(std::vector<Point>& points, double /*toler*/) {
    if(points.size() < 2) return;
    std::vector<double> xVals(points.size()), yVals(points.size());
    for(size_t i = 0; i < points.size(); i++){
        xVals[i] = points[i].x;
        yVals[i] = points[i].y;
    }
    auto xFreq = dft(xVals);
    auto yFreq = dft(yVals);

    // Keep low 20% of frequencies
    size_t cutoff = (size_t)(xFreq.size() * 0.2);
    for(size_t i = cutoff; i < xFreq.size(); i++) {
        xFreq[i] = 0;
        yFreq[i] = 0;
    }

    auto xFiltered = idft(xFreq);
    auto yFiltered = idft(yFreq);
    for(size_t i = 0; i < points.size(); i++){
        points[i].x = xFiltered[i];
        points[i].y = yFiltered[i];
    }
}

// Determine arc direction via cross product of center->start and center->end
static ArcDirection findArcDirection(const Point& center, const Point& start, const Point& end) {
    double cross = (start.x - center.x)*(end.y - center.y) - (start.y - center.y)*(end.x - center.x);
    return (cross < 0.0) ? ArcDirection::CW : ArcDirection::CCW;
}

// Convert simplified points to segments
std::vector<Segment> buildSimplifiedSegments(const std::vector<Point>& points, double toler) {
    std::vector<Segment> result;
    if (points.size() < 2) return result;

    size_t i = 0;
    while (i + 2 < points.size()) {
        double curv = computeCurvature(points[i], points[i+1], points[i+2]);
        Segment seg;
        seg.start = points[i];
        seg.end   = points[i+1];

        if(std::abs(curv) < toler) {
            seg.type = SegmentType::LINE;
            seg.center = {0,0};
            seg.radius = 0;
            seg.direction = ArcDirection::CW; // unused
        } else {
            seg.type = SegmentType::ARC;
            seg.center = points[i+1]; // naive
            seg.radius = dist(points[i], points[i+1]);
            seg.direction = findArcDirection(seg.center, seg.start, seg.end);
        }
        result.push_back(seg);
        i++;
    }

    // Handle leftover if any
    if(i + 1 < points.size()) {
        Segment seg;
        seg.type = SegmentType::LINE;
        seg.start = points[i];
        seg.end = points.back();
        seg.center = {0,0};
        seg.radius = 0;
        seg.direction = ArcDirection::CW; // unused
        result.push_back(seg);
    }

    return result;
}

// Read CSV
std::vector<Segment> readSegmentsFromCSV(const std::string& filename) {
    std::vector<Segment> segments;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string typeStr;
        std::getline(ss, typeStr, ',');

        Segment seg;
        if (typeStr == "LINE") {
            seg.type = SegmentType::LINE;
            seg.direction = ArcDirection::CW; // not used for lines
        } else if (typeStr == "ARC") {
            seg.type = SegmentType::ARC;
        } else {
            continue; 
        }

        ss >> seg.start.x; ss.ignore();
        ss >> seg.start.y; ss.ignore();
        ss >> seg.end.x;   ss.ignore();
        ss >> seg.end.y;
        if (seg.type == SegmentType::ARC) {
            ss.ignore();
            ss >> seg.center.x; ss.ignore();
            ss >> seg.center.y; ss.ignore();
            ss >> seg.radius;
            // Attempt to deduce direction if missing from input 
            seg.direction = findArcDirection(seg.center, seg.start, seg.end);
        }
        segments.push_back(seg);
    }
    return segments;
}

// Write final segments to CSV
void writeSegmentsToCSV(const std::vector<Segment>& segments, const std::string& outFile) {
    std::ofstream out(outFile);
    if(!out.is_open()) {
        throw std::runtime_error("Cannot open output file: " + outFile);
    }
    out << std::fixed << std::setprecision(2);
    for(const auto& seg : segments) {
        if(seg.type == SegmentType::LINE) {
            out << "LINE,"
                << seg.start.x << "," << seg.start.y << ","
                << seg.end.x   << "," << seg.end.y   << "\n";
        } else {
            // ARC
            out << "ARC,"
                << seg.start.x << "," << seg.start.y << ","
                << seg.end.x   << "," << seg.end.y   << ","
                << seg.center.x<< "," << seg.center.y<< ","
                << seg.radius  << "\n";
        }
    }
    out.close();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.csv> <tolerance>\n";
        return 1;
    }

    try {
        std::string filename = argv[1];
        double tolerance = std::stod(argv[2]);

        // Read input
        std::vector<Segment> original = readSegmentsFromCSV(filename);
        std::cout << "Original segments: " << original.size() << std::endl;

        // Flatten points
        std::vector<Point> points;
        for(const auto& seg : original){
            points.push_back(seg.start);
        }
        if(!original.empty()) {
            points.push_back(original.back().end);
        }

        // Optional FFT filter
        applyFFTFilter(points, tolerance);

        // RDP simplify
        points = rdpSimplify(points, tolerance);

        // Build final segments
        auto simplified = buildSimplifiedSegments(points, tolerance);
        std::cout << "Simplified segments: " << simplified.size() << std::endl;

        // Write output CSV
        std::string outFile = filename.substr(0, filename.find_last_of('.')) + "_simplified.csv";
        writeSegmentsToCSV(simplified, outFile);
        std::cout << "Wrote simplified segments to " << outFile << std::endl;

    } catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

