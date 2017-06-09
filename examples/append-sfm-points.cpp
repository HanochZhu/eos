#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include "rapidxml-1.13/rapidxml_utils.hpp"

#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;
using std::string;

typedef rapidxml::xml_node<> xmlnode;
typedef rapidxml::xml_attribute<> xmlattr;
typedef rapidxml::xml_document<> xmldoc;

class Point {
  public:
    float x;
    float y;

    Point(float x, float y) {
      this->x = x;
      this->y = y;
    }
};
class Image {
  public:
    string fullpath;
    string path;
    LandmarkCollection<cv::Vec2f> landmarks;
    vector<Point> sfmPoints;

    Image(string fp, string p, LandmarkCollection<cv::Vec2f> l) 
    {
      fullpath = fp;
      path = p;
      landmarks = l;
    }
};



vector<int> read_sfm_indices(std::string filename) {
  std::ifstream infile(filename);
	if (!infile.is_open()) {
		throw std::runtime_error(string("Could not open SFM indices file: " + filename));
	}
  vector<int> sfmIndices;
  int sfmIndex;
  cout << "Using SFM indices: [ ";
  while(infile >> sfmIndex) {
    cout << sfmIndex << ' ';
    sfmIndices.push_back(sfmIndex);
  }
  cout << "]\n";

  return sfmIndices;
}

void read_training_xml(std::string filename, xmldoc* doc) {
  rapidxml::file<> xmlFile(filename.c_str());
  doc->parse<rapidxml::parse_full>(xmlFile.data());
}

/**
 * Given an XML node containing an image, return an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @return An ordered vector with the 68 ibug landmarks.
 */
LandmarkCollection<cv::Vec2f> training_xml_to_landmarks(xmlnode *image)
{
	using cv::Vec2f;
	using std::string;
	LandmarkCollection<Vec2f> landmarks;
	landmarks.reserve(68);

  if (image->first_attribute("file") == 0) {
    cout << "well shit...\n";
    return landmarks;
  }
  string imagePath = image->first_attribute("file")->value();

  xmlnode *box = image->first_node();
  int ibugId = 1;
  for(xmlnode *p = box->first_node(); p != 0; p = p->next_sibling()) {
    string x = p->first_attribute("x")->value();
    string y = p->first_attribute("y")->value();

    if (x == "" || y == "")
      throw std::runtime_error(string("Landmark format error while parsing image: " + imagePath));

    Landmark<Vec2f> landmark;
    landmark.name = std::to_string(ibugId);
    landmark.coordinates[0] = stof(x); // x and y need to be floats
    landmark.coordinates[1] = stof(y);
    landmarks.emplace_back(landmark);

    ++ibugId;
  }

  if (ibugId < 69)
    throw std::runtime_error(string("Not enough landmark points defined for image: " + imagePath));

	return landmarks;
};

/**
 * Retrieve sfm points from the image
 *
 *
 * @param[in] mesh The mesh to draw.
 * @param[in] modelview Model-view matrix to draw the mesh.
 * @param[in] projection Projection matrix to draw the mesh.
 * @param[in] viewport Viewport to draw the mesh.
 * @param[in] sfmIndices SFM indices vector
 */
vector<Point> get_sfm_points(const core::Mesh& mesh, glm::mat4x4 modelview, glm::mat4x4 projection, glm::vec4 viewport, vector<int> sfmIndices)
{
  vector<Point> points;

  int i = 0;
	for (const auto& triangle : mesh.tvi)
	{
		const auto p1 = glm::project({ mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2] }, modelview, projection, viewport);
    if (std::find(sfmIndices.begin(), sfmIndices.end(), i) != sfmIndices.end()) {
      points.push_back(Point(p1.x, p1.y));
    }

    ++i;
	}

  return points;
};

int main(int argc, char *argv[])
{
  using std::to_string;

	fs::path modelfile, isomapfile, imagefile, landmarksfile, mappingsfile, contourfile, edgetopologyfile, blendshapesfile, outputfile, indexfile, trainingxml, trainingdir;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("indices,s", po::value<fs::path>(&indexfile)->required()->default_value("../share/sfm_indices"),
				"set of SFM indices")
			("trainingdir,d", po::value<fs::path>(&trainingdir)->required()->default_value("data/ibug_300W_large_face_landmark_dataset"),
				"base directory containing training dataset")
			("trainingxml,t", po::value<fs::path>(&trainingxml)->required()->default_value("data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml"),
				"training data XML file")
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
				"a Morphable Model stored as cereal BinaryArchive")
			("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
				"landmark identifier to model vertex number mapping")
			("model-contour,c", po::value<fs::path>(&contourfile)->required()->default_value("../share/model_contours.json"),
				"file with model contour indices")
			("edge-topology,e", po::value<fs::path>(&edgetopologyfile)->required()->default_value("../share/sfm_3448_edge_topology.json"),
				"file with model's precomputed edge topology")
			("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value("../share/expression_blendshapes_3448.bin"),
				"file with blendshapes")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("out"),
				"basename for the output rendering and obj files")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: fit-model [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	} catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_FAILURE;
	}

  // load SFM indices file
  vector<int> sfmIndices;
	try {
    sfmIndices = read_sfm_indices(indexfile.string());
	} catch (const std::runtime_error& e) {
		cout << "Error reading the SFM indices file: " << e.what() << endl;
		return EXIT_FAILURE;
	}

  // load training XML file
  xmldoc trainingXMLDoc;
	try {
    read_training_xml(trainingxml.string(), &trainingXMLDoc);
	} catch (const std::runtime_error& e) {
		cout << "Error reading training xml file: " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the image, landmarks, LandmarkMapper and the Morphable Model:
	morphablemodel::MorphableModel morphable_model;
	try {
		morphable_model = morphablemodel::load_model(modelfile.string());
	} catch (const std::runtime_error& e) {
		cout << "Error loading the Morphable Model: " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// The landmark mapper is used to map ibug landmark identifiers to vertex ids:
	core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

	// The expression blendshapes:
	vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile.string());

	// These two are used to fit the front-facing contour to the ibug contour landmarks:
	fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());
	fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());

	// The edge topology is used to speed up computation of the occluding face contour fitting:
	morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile.string());

	// Draw the loaded landmarks:
  /*
	Mat outimg = mat.clone();
	for (auto&& lm : landmarks) {
		cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
	}
  */
  vector<Image> images;
  xmlnode *imagesNode = trainingXMLDoc.first_node("dataset")->first_node("images");
  for(xmlnode *image = imagesNode->first_node("image"); image != 0; image = image->next_sibling("image")) {
    string imageSubPath = image->first_attribute()->value();
    string imagePath = trainingdir.string() + "/" + imageSubPath;

    LandmarkCollection<cv::Vec2f> landmarks;
    try {
      landmarks = training_xml_to_landmarks(image);
    } catch (const std::runtime_error& e) {
      cout << "Error reading the landmarks for " << imagePath << ": " << e.what() << endl;
      return EXIT_FAILURE;
    }

    images.push_back(Image(imagePath, imageSubPath, landmarks));

  }

  //cout << "Loading and constructing meshes\n";
  for(auto&& image : images) {
    Mat mat = cv::imread(image.fullpath);

    // Fit the model, get back a mesh and the pose:
    core::Mesh mesh;
    fitting::RenderingParameters rendering_params;
    std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(morphable_model, blendshapes, image.landmarks, landmark_mapper, mat.cols, mat.rows, edge_topology, ibug_contour, model_contour, 50, boost::none, 30.0f);

    image.sfmPoints = get_sfm_points(mesh, rendering_params.get_modelview(), rendering_params.get_projection(), fitting::get_opencv_viewport(mat.cols, mat.rows), sfmIndices);

    string csvline = image.path;
    // append 68 points
    for(auto&& l : image.landmarks)
      csvline = csvline + "," + to_string(l.coordinates[0]) + "," + to_string(l.coordinates[1]);
    // append sfm points
    for(auto&& sfmPoint : image.sfmPoints)
      csvline = csvline + "," + to_string(sfmPoint.x) + "," + to_string(sfmPoint.y);
    cout << csvline << '\n';
    break;
  }

	cout << "FIN" << outputfile.stem().stem() << "." << endl;

	return EXIT_SUCCESS;
}
