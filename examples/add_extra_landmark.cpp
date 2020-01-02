//
//  main.cpp
//  TrainSetBatch
//
//  Created by zhuhongqiang on 2019/8/7.
//

#include "eos/core/Image.hpp"
#include "eos/core/image/opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/core/write_obj.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/draw_utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/cpp17/optional.hpp"

#include "Eigen/Core"

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <algorithm>
#include <string>
#include <iostream>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

vector<int> read_sfm_indices(std::string filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error(string("Could not open SFM indices file: " + filename));
    }
    vector<int> sfmIndices;
    int sfmIndex;
    while(infile >> sfmIndex) {
        sfmIndices.push_back(sfmIndex);
    }
    
    return sfmIndices;
}

//draw extra point of face
//in batch process, we mark extra point of face,
//and store in marked file
void save_extra_point(string filepath,const core::Mesh& mesh, glm::mat4x4 modelview, glm::mat4x4 projection, glm::vec4 viewport, vector<int> sfmIndices, cv::Scalar colour = cv::Scalar(0, 255, 0, 255)){
    
    int i = 0;
    size_t temp_indice_index = 0;
    cout<<"append new point in file "<<filepath<<endl;
    std::string appendTxt;
    
    for (const auto& triangle : mesh.tvi)
    {

        //indice refer to face index of mesh
        const auto p1 = glm::project({ mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2] }, modelview, projection, viewport);

        
        if (std::find(sfmIndices.begin(), sfmIndices.end(), i) != sfmIndices.end()) {
            //cout << "point " << p1.x << " " << p1.y << " index: " << i << '\n';
            ++ temp_indice_index;
            
            appendTxt += std::to_string(p1.x) + " , " + std::to_string(p1.y) + "\n";
            //append extra point info to txtfile
            
        }
        if (sfmIndices.size() <= temp_indice_index) {
            break;
        }
        ++i;
    }
    std::ofstream ofs(filepath,std::ofstream::app);
    ofs<<appendTxt;
    ofs.close();
}


int main(int argc, const char * argv[]) {
    
    std::map<string, string> image_mark_file_map;
    std::vector<string> lost_image_list;
    //default value
    int max_file_size = 2330;
    int max_file_line_count = 197;
//    string modelfile = "../share/sfm_shape_3448.bin";
//    string indexfile = "../share/sfm_indices";
//    string edgetopologyfile = "../share/sfm_3448_edge_topology.json";
//    string mappingsfile = "../share/ibug_to_sfm.txt";
//    string contourfile = "../share/sfm_model_contours.json";
//    string blendshapesfile = "../share/expression_blendshapes_3448.bin";
    //string outputbasename = "out";
    
//    string abstract_image_path = "../../../../faces/train/";
//    string abstract_pts_file_path = "../../../../faces/data/";
//    string abstract_annotation_file_path = "../../../../faces/annotationt/";
    if (argc < 5) {
        cout<<"check your input, if it contains image_path(the first arg),"<<
            "pts_path(the second arg),"<<
            "annotation_path(the third arg),"<<
            "share_path(the fourth arg)"<<endl;
        return 0;
    }
    string abstract_image_path = argv[1];
    string abstract_pts_file_path = argv[2];
    string abstract_annotation_file_path = argv[3];
    string abstract_share_path = argv[4];
    
    string modelfile = abstract_share_path + "sfm_shape_3448.bin";
    string indexfile = abstract_share_path + "sfm_indices";
    string edgetopologyfile = abstract_share_path + "sfm_3448_edge_topology.json";
    string mappingsfile = abstract_share_path + "ibug_to_sfm.txt";
    string contourfile = abstract_share_path + "sfm_model_contours.json";
    string blendshapesfile = abstract_share_path + "expression_blendshapes_3448.bin";

    
    
    //load indice file
    vector<int> sfmIndices;
    try {
        sfmIndices = read_sfm_indices(indexfile);
    } catch (const std::runtime_error& e) {
        cout << "Error reading the SFM indices file: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    
    //load raw mofiable model
    morphablemodel::MorphableModel morphable_model;
    try
    {
        morphable_model = morphablemodel::load_model(modelfile);
    } catch (const std::runtime_error& e)
    {
        cout << "Error loading the Morphable Model: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    
    core::LandmarkMapper landmark_mapper;
    try
    {
        landmark_mapper = core::LandmarkMapper(mappingsfile);
    } catch (const std::exception& e)
    {
        cout << "Error loading the landmark mappings: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    
    
    // The expression blendshapes:
    const vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile);
    
    const fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile);
    
    // The edge topology is used to speed up computation of the occluding face contour fitting:
    const morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);
    
    
    
    for (int ifile = 1; ifile <= max_file_size; ++ifile) {
        //get & store image path and landmark file path
        string mark_point_file_path = abstract_annotation_file_path + std::to_string(ifile) + ".txt";
        
        // cout<<"open file " << mark_point_file_path<<endl;
        
        std::ifstream tempf(mark_point_file_path,std::ifstream::in);
        if (!tempf.is_open()) {
            cout<<"can not open file " << mark_point_file_path<<endl;
            continue;
        }
        
        string image_name;
        tempf>>image_name;
        cout<<image_name<<endl;
        string imagefile = abstract_image_path + image_name + ".jpg";
        cout<<"cur load image" <<imagefile<<std::endl;
        int line_num  = std::count(std::istreambuf_iterator<char>(tempf),
                   std::istreambuf_iterator<char>(), '\n');
        //if the max number of file is large than 194
        //the file is already be used.
        if (line_num  > max_file_line_count) {
            cout<<imagefile<<" is already loaded"<<endl;
            continue;
        }
        tempf.close();
        
        //store image name and landmark file in a map container
        //exp : <10100000:1.txt>
        image_mark_file_map.insert(std::pair<std::string, std::string>(image_name,mark_point_file_path));

        //landmark file path with 68 points
        string landmarksfile = abstract_pts_file_path + image_name + ".pts";

//        vector<int> sfmIndices;
//        try {
//            sfmIndices = read_sfm_indices(indexfile);
//        } catch (const std::runtime_error& e) {
//            cout << "Error reading the SFM indices file: " << e.what() << endl;
//            return EXIT_FAILURE;
//        }
        
        // Load the image, landmarks, LandmarkMapper and the Morphable Model:
        Mat image = cv::imread(imagefile);
        LandmarkCollection<Eigen::Vector2f> landmarks;
        
        // landmarksfile value
        try
        {
            landmarks = core::read_pts_landmarks(landmarksfile);
            if (landmarks.size() <= 0) {
                lost_image_list.push_back(imagefile);
                continue;
            }
        } catch (const std::runtime_error& e)
        {
            cout << "Error reading the landmarks: " << e.what() << endl;
            lost_image_list.push_back(e.what());
            continue;
        }
        morphablemodel::MorphableModel tmorphable_model = morphable_model;
//        try
//        {
//            morphable_model = morphablemodel::load_model(modelfile);
//        } catch (const std::runtime_error& e)
//        {
//            cout << "Error loading the Morphable Model: " << e.what() << endl;
//            return EXIT_FAILURE;
//        }
        // The landmark mapper is used to map 2D landmark points (e.g. from the ibug scheme) to vertex ids:
        core::LandmarkMapper tlandmark_mapper = landmark_mapper;
//        try
//        {
//            landmark_mapper = core::LandmarkMapper(mappingsfile);
//        } catch (const std::exception& e)
//        {
//            cout << "Error loading the landmark mappings: " << e.what() << endl;
//            return EXIT_FAILURE;
//        }
//
//        // The expression blendshapes:
//        const vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile);
        
        morphablemodel::MorphableModel morphable_model_with_expressions(
                                                                        tmorphable_model.get_shape_model(), blendshapes, tmorphable_model.get_color_model(), cpp17::nullopt,
                                                                        tmorphable_model.get_texture_coordinates());
        
        // These two are used to fit the front-facing contour to the ibug contour landmarks:
        const fitting::ModelContour model_contour =
                                contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile);
        
//        const fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile);
//
//        // The edge topology is used to speed up computation of the occluding face contour fitting:
//        const morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);
        
        // Draw the loaded landmarks:
//        Mat outimg = image.clone();
//        for (auto&& lm : landmarks)
//        {
//            cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
//                        cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), {255, 0, 0});
//        }
        
        // Fit the model, get back a mesh and the pose:
        
        core::Mesh mesh;
        fitting::RenderingParameters rendering_params;
        std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(
                                                                    morphable_model_with_expressions, landmarks, tlandmark_mapper, image.cols, image.rows, edge_topology,
                                                                    ibug_contour, model_contour, 5, cpp17::nullopt, 30.0f);
        
        // The 3D head pose can be recovered as follows:
        float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
        // and similarly for pitch and roll.
        
        // Extract the texture from the image using given mesh and camera parameters:
        const Eigen::Matrix<float, 3, 4> affine_from_ortho =
                                fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
        const core::Image4u isomap =
                                render::extract_texture(mesh, affine_from_ortho, core::from_mat(image), true);
        
        //save extra face landmark
        save_extra_point(mark_point_file_path, mesh, rendering_params.get_modelview(), rendering_params.get_projection(), fitting::get_opencv_viewport(image.cols, image.rows), sfmIndices);
        

//        cout << "Finished fitting and wrote result mesh and isomap to files with basename "
//             << outputfile.stem().stem() << "." << endl;
            cout << "Finished fitting and wrote result mesh and isomap to files with basename "<<mark_point_file_path<< endl;

            
    }
    
    for (auto tname : lost_image_list) {
        cout<<tname<<endl;
    }
    
//    string modelfile, isomapfile, imagefile, landmarksfile, mappingsfile, contourfile, edgetopologyfile,
//        blendshapesfile, outputbasename,indexfile;
//    try
//    {
//        po::options_description desc("Allowed options");
//        // clang-format off
//        desc.add_options()
//        ("help,h", "display the help message")
//        ("model,m", po::value<string>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
//         "a Morphable Model stored as cereal BinaryArchive")
//        ("indice,s", po::value<string>(&indexfile)->required()->default_value("../share/sfm_indices"),
//         "set indice file")
//        ("image,i", po::value<string>(&imagefile)->required()->default_value("data/image_0010.png"),
//         "an input image")
//        ("landmarks,l", po::value<string>(&landmarksfile)->required()->default_value("data/image_0010.pts"),
//         "2D landmarks for the image, in ibug .pts format")
//        ("mapping,p", po::value<string>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
//         "landmark identifier to model vertex number mapping")
//        ("model-contour,c", po::value<string>(&contourfile)->required()->default_value("../share/sfm_model_contours.json"),
//         "file with model contour indices")
//        ("edge-topology,e", po::value<string>(&edgetopologyfile)->required()->default_value("../share/sfm_3448_edge_topology.json"),
//         "file with model's precomputed edge topology")
//        ("blendshapes,b", po::value<string>(&blendshapesfile)->required()->default_value("../share/expression_blendshapes_3448.bin"),
//         "file with blendshapes")
//        ("output,o", po::value<string>(&outputbasename)->required()->default_value("out"),
//         "basename for the output rendering and obj files");
//        // clang-format on
//        po::variables_map vm;
//        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
//        if (vm.count("help"))
//        {
//            cout << "Usage: fit-model [options]" << endl;
//            cout << desc;
//            return EXIT_SUCCESS;
//        }
//        po::notify(vm);
//    } catch (const po::error& e)
//    {
//        cout << "Error while parsing command-line arguments: " << e.what() << endl;
//        cout << "Use --help to display a list of options." << endl;
//        return EXIT_FAILURE;
//    }
    
    
    
    return EXIT_SUCCESS;
}
