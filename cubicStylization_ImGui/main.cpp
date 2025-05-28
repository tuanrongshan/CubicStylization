#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/snap_points.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <normalize_unitbox.h>
#include <cube_style_data.h>
#include <cube_style_precomputation.h>
#include <cube_style_precomputation_quad.h>
#include "cube_style_single_iteration.h"
#include <get_bounding_box.h>

#include <ctime>
#include <vector>
#include <iostream>
#include <fstream>

#ifndef MESH_PATH
#  define MESH_PATH "../../meshes/"
#endif

#ifndef OUTPUT_PATH
#define OUTPUT_PATH "../"
#endif


static bool useQuadMesh = false;
inline void add_axes(
    igl::opengl::ViewerData &data,
    const Eigen::Vector3d &origin,
    double scale = 100.0)
{
    using Row3 = Eigen::RowVector3d;

    Row3 O = origin.transpose();

    Row3 X1 = O + Row3(scale, 0, 0);
    Row3 Y1 = O + Row3(0, scale, 0);
    Row3 Z1 = O + Row3(0, 0, scale);

    data.add_edges(O, X1, Row3(1.0, 0.0, 0.0)); 
    data.add_edges(O, Y1, Row3(0.0, 1.0, 0.0)); 
    data.add_edges(O, Z1, Row3(0.0, 0.0, 1.0)); 
}

struct Mode
{
    Eigen::MatrixXd CV; // point constraint
    bool place_constraints = true;
    bool polyhedral = false;
    int numCV = 0;
    Eigen::MatrixXi CV_row_col; 
} state;

static float g_uv_scale = 3.0f;
static int texture_mode = 0;
int main(int , char**)
{
    using namespace Eigen; 
    using namespace std;
    namespace iglm = igl::opengl::glfw::imgui; 

    float anim_t = 0.0f; // interpolation for polyhedral

    vector<string> model_names = {
        "beetle.obj",
        "bumpy.obj",
        "cactus.obj",
        "kleinBottle.obj",
        "ogre.obj",
        "spot.obj",
        "bob.obj",
        "bunny.obj",
        "horse.obj",
        "lilium.obj",
        "rockerArm.obj",
        "only_quad_sphere.obj"
    };
    int current_model = 0;

    bool orthographic  = false;
    bool face_based    = true;
    bool show_tex      = false;
    bool show_overlay  = true;
    bool wireframe     = false;
    bool fill_faces    = true;
    char outputName[128] = "output";

    RowVector3d mesh_color(149./255, 217./255, 244./255); 
    Vector3f    bg_color (208./255, 237./255, 227./255); 

    igl::opengl::glfw::Viewer           viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    MatrixXd V,U,VO, TC, U1;
    MatrixXi F, FT, F1;
    cube_style_data data; 
    data.lambda = 2e-1;
    int maxCV = 100;
    state.CV_row_col.resize(maxCV,2); // assume nor more than 100 constraints
    
    

    double theta_x = 10.0; 
	double theta_y = 10.0; 
	double theta_z = 10.0; 
	Matrix3d Rx, iRx, Ry, iRy, Rz, iRz;
    {
        Rx << 
            1., 0., 0.,
            0., cos(theta_x/180.*3.14), -sin(theta_x/180.*3.14),
            0., sin(theta_x/180.*3.14), cos(theta_x/180.*3.14);
        Ry <<
            cos(theta_y/180.*3.14), 0., sin(theta_y/180.*3.14),
            0., 1., 0.,
            -sin(theta_y/180.*3.14), 0, cos(theta_y/180.*3.14);
        Rz <<
            cos(theta_z/180.*3.14), -sin(theta_z/180.*3.14), 0.,
            sin(theta_z/180.*3.14), cos(theta_z/180.*3.14), 0.,
            0., 0., 1;
    }

    const RowVector3d red (250./255,114./255,104./255);
    const RowVector3d blue(149./255,217./255,244./255);
    const RowVector3d gray(200./255,200./255,200./255);

    auto ensure_uv = [&](){
        if(TC.rows()!=0) return;
        RowVector3d mn = V.colwise().minCoeff();
        RowVector3d mx = V.colwise().maxCoeff();
        TC.resize(V.rows(),2);
        TC.col(0) = g_uv_scale * (V.col(0).array()-mn[0])/(mx[0]-mn[0]);
        TC.col(1) = g_uv_scale * (V.col(1).array()-mn[1])/(mx[1]-mn[1]);
        FT = F;
    };
    auto load_mesh = [&](int idx)
    {
        anim_t = 0.0;
        state.polyhedral = false;
        igl::readOBJ( string(MESH_PATH)+model_names[idx], V, F );
        igl::readOBJ(string(MESH_PATH)+(string)"poly_"+model_names[idx], U1, F1);
        // if(F1.cols() == 4) {
        //     Eigen::MatrixXi F_tr(F1.rows() * 2, 3);
        //     for(int i = 0; i < F1.rows(); ++i) {
        //         int v0 = F1(i, 0), v1 = F1(i, 1), v2 = F1(i, 2), v3 = F1(i, 3);
        //         // fan triangulation
        //         F_tr.row(2*i)     << v0, v1, v2;
        //         F_tr.row(2*i + 1) << v0, v2, v3;
        //     }
        //     F1 = F_tr;
        // }
        normalize_unitbox(V);
        normalize_unitbox(U1);
        V.rowwise() -= V.colwise().mean();
        U1.rowwise() -= U1.colwise().mean();
        U  = VO = V;
        state.CV.resize(0,3); state.numCV = 0;
        state.place_constraints = true;
        viewer.data().grid_texture();   
    };

    const auto & resetCV = [&]()
    {
        for (int ii=0; ii<state.numCV; ii++)
        {
            int fid = state.CV_row_col(ii,0);
            int c   = state.CV_row_col(ii,1);
            RowVector3d new_c = V.row(F(fid,c));
            state.CV.row(ii) = new_c;
        }
    };

    auto precompute = [&](){
    if(state.CV.rows()==0){
        state.CV.resize(1,3);
        state.CV.row(0)          = V.row( F(0,0) );
        state.CV_row_col.row(0) << 0,0;
        state.numCV = 1;
    }
    igl::snap_points(state.CV,V,data.b);
    data.bc.resize(data.b.size(),3);
    for(int i=0;i<data.b.size();++i) data.bc.row(i)=V.row(data.b(i));
        U = V;
        cube_style_precomputation(V,F,data);
    };
    const auto draw = [&](){
        viewer.data().clear();
        viewer.data().face_based = true;

        if(state.polyhedral)
        {
            Eigen::MatrixXd Vframe = (1.0 - anim_t) * (state.place_constraints? V: U) + anim_t * U1;
            
            viewer.data().set_mesh(Vframe, F1);
            viewer.data().set_colors(mesh_color);
            viewer.data().compute_normals();
            MatrixXd V_box;
            MatrixXi E_box;
            get_bounding_box(V, V_box, E_box);
            viewer.data().add_points(V_box, red);
            for (unsigned i=0;i<E_box.rows(); ++i)
                viewer.data().add_edges(V_box.row(E_box(i,0)),V_box.row(E_box(i,1)),red);
            
            if(anim_t < 1.0f) anim_t += 0.01f;
        }else if(state.place_constraints){
            viewer.data().set_mesh(V,F);
            viewer.data().set_colors(mesh_color);          // ← 原本的 gray/blue 換成這
            viewer.data().set_points(state.CV, red);
            MatrixXd V_box;
            MatrixXi E_box;
            get_bounding_box(V, V_box, E_box);
            viewer.data().add_points(V_box, red);
            for (unsigned i=0;i<E_box.rows(); ++i)
                viewer.data().add_edges(V_box.row(E_box(i,0)),V_box.row(E_box(i,1)),red);
        }else{
            cube_style_single_iteration(V,U,data);
            viewer.data().set_mesh(U,F);
            viewer.data().set_colors(mesh_color);          // ← 同上
            viewer.data().set_points(state.CV, red);
            MatrixXd V_box;
            MatrixXi E_box;
            get_bounding_box(V, V_box, E_box);
            viewer.data().add_points(V_box, red);
            for (unsigned i=0;i<E_box.rows(); ++i)
                viewer.data().add_edges(V_box.row(E_box(i,0)),V_box.row(E_box(i,1)),red);
        }
        viewer.data().set_points(state.CV,red);
        if(viewer.data().show_texture)
        {
            ensure_uv();
            viewer.data().set_uv(TC, FT);

            if(texture_mode == 0) {
                viewer.data().grid_texture(); 
            }
            else if(texture_mode == 1) {
                int size = 128;
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(size, size), G(size, size), B(size, size);
                for(int i = 0; i < size; ++i) {
                    for(int j = 0; j < size; ++j) {
                        if((j / 16) % 2 == 0) {
                            R(i,j) = 255; G(i,j) = 255; B(i,j) = 255; 
                        } else {
                            R(i,j) = 0; G(i,j) = 0; B(i,j) = 0; 
                        }
                    }
                }
                viewer.data().set_texture(R, G, B);
            }else if(texture_mode == 2) {
                int size = 128;
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(size, size), G(size, size), B(size, size);
                for(int i = 0; i < size; ++i) {
                    for(int j = 0; j < size; ++j) {
                        if((i / 16) % 2 == 0) {
                            R(i,j) = 255; G(i,j) = 255; B(i,j) = 255; 
                        } else {
                            R(i,j) = 0; G(i,j) = 0; B(i,j) = 0; 
                        }
                    }
                }
                viewer.data().set_texture(R, G, B);
            }else if(texture_mode == 3) {
                int size = 256;
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(size, size), G(size, size), B(size, size);
                float frequency = 0.1; // 控制波紋密度

                for(int i = 0; i < size; ++i) {
                    for(int j = 0; j < size; ++j) {
                        float dx = i - size / 2;
                        float dy = j - size / 2;
                        float dist = sqrt(dx * dx + dy * dy);
                        float wave = 0.5 * (sin(dist * frequency) + 1.0); // 值域在 [0,1]

                        unsigned char color = static_cast<unsigned char>(wave * 255);
                        R(i,j) = color;
                        G(i,j) = color;
                        B(i,j) = color;
                    }
                }

                viewer.data().set_texture(R, G, B);
            }
        }

        // 軸
        Eigen::Vector3d bbmin = V.colwise().minCoeff();
        Eigen::Vector3d bbmax = V.colwise().maxCoeff();
        double axis_len = 0.40 * (bbmax - bbmin).norm(); 

        viewer.data().line_width = 5.0f;           
        add_axes(viewer.data(), Eigen::Vector3d::Zero(), axis_len);
    };

    menu.callback_draw_viewer_window = [&](){
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x-220, 0), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(220, 400), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Viewer",nullptr,ImGuiWindowFlags_NoSavedSettings);
        

        if(ImGui::CollapsingHeader("Viewing", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::DragFloat("Zoom",&viewer.core().camera_zoom,0.005f,0.1f,5.f,"%.2f");
            orthographic = viewer.core().orthographic;
            if(ImGui::Checkbox("Orthographic", &orthographic))
                viewer.core().orthographic = orthographic;
        }

        if(ImGui::CollapsingHeader("Draw", ImGuiTreeNodeFlags_DefaultOpen))
        {
            show_tex = viewer.data().show_texture;
            if(ImGui::Checkbox("Show texture", &show_tex))
                viewer.data().show_texture = show_tex;

            if(show_tex)
            {
                if(ImGui::SliderFloat("UV scale", &g_uv_scale, 0.5f, 10.0f, "%.2f"))
                {
                    TC.resize(0,2);        
                    draw();             
                }
                
                if (ImGui::RadioButton("checkerboard", texture_mode == 0)) {
                    texture_mode = 0;
                    draw();  
                }
                if (ImGui::RadioButton("Horizontal stripes", texture_mode == 1)){
                    texture_mode = 1;
                    draw();  
                } 
                if (ImGui::RadioButton("Vertical stripes", texture_mode == 2)){
                    texture_mode = 2;
                    draw(); 
                }
                if (ImGui::RadioButton("Wave pattern", texture_mode == 3)){
                    texture_mode = 3;
                    draw(); 
                }
            }
            
            show_overlay = viewer.data().show_overlay;
            if(ImGui::Checkbox("Show overlay", &show_overlay))
                viewer.data().show_overlay = show_overlay;

            wireframe = viewer.data().show_lines;
            if(ImGui::Checkbox("Wireframe", &wireframe))
                viewer.data().show_lines = wireframe;

            fill_faces = viewer.data().show_faces;
            if(ImGui::Checkbox("Fill faces", &fill_faces))
                viewer.data().show_faces = fill_faces;
        }
        ImGui::Separator();
        {
            ImGui::Text("Instructions");
            ImGui::BulletText("[click]  place constrained points");
            ImGui::BulletText("[space]  change mode");
            ImGui::BulletText("R          reset ");
            ImGui::BulletText("</>      decrease/increase lambda");
            ImGui::BulletText("Q/W    rotate x-axis");
            ImGui::BulletText("A/S      rotate y-axis");
            ImGui::BulletText("Z/X      rotate z-axis");
            ImGui::BulletText("L          show edges");
            ImGui::Text(" ");
            ImGui::Checkbox("Use quadrilateral mesh", &useQuadMesh);
        }
        ImGui::End();
    };

    menu.callback_draw_custom_window = [&](){
        {
            ImGui::SetNextWindowPos(ImVec2(0.f * menu.menu_scaling(),0),ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(220, 400), ImGuiSetCond_FirstUseEver);
            ImGui::Begin(
                "Cubic Stylization", nullptr,
                ImGuiWindowFlags_NoSavedSettings
            );
        }
        {
            ImGui::DragScalar("lambda",ImGuiDataType_Double,&data.lambda,2e-1,0,0,"%.2e");
            ImGui::Separator();
            ImGui::Text("Models:");
            for(int i=0;i<(int)model_names.size();++i)
                if(ImGui::Selectable(model_names[i].c_str(), i==current_model))
                { 
                    current_model=i; 
                    load_mesh(i); 
                    draw(); 
                }
            ImGui::Separator();
            ImGui::ColorEdit3("Background", bg_color.data());
            if(ImGui::IsItemDeactivatedAfterEdit())
            {
                viewer.core().background_color << bg_color[0], bg_color[1], bg_color[2], 1.0f;
            }
    
            static float mc[3];
            mc[0] = static_cast<float>(mesh_color[0]);
            mc[1] = static_cast<float>(mesh_color[1]);
            mc[2] = static_cast<float>(mesh_color[2]);
    
            if(ImGui::ColorEdit3("Mesh color", mc))
            {
                mesh_color << mc[0], mc[1], mc[2];
                draw();                     
            }
        }
        {
            // output file name
            ImGui::InputText(".obj", outputName, IM_ARRAYSIZE(outputName));
        }
        if (ImGui::Button("save output mesh", ImVec2(-1,0)))
		{
			string outputFile = OUTPUT_PATH;
            outputFile.append("cubic_");
            outputFile.append(outputName);
            outputFile.append(".obj");
			igl::writeOBJ(outputFile, U, F);

            string inputFile = OUTPUT_PATH;
            inputFile.append("input_");
            inputFile.append(outputName);
            inputFile.append(".obj");
			igl::writeOBJ(inputFile, V, F);
		}
        ImGui::End();
    };

    // recompute state.CV function
    

    // draw function
    // const auto & draw = [&]()
    // {
    //     const Eigen::RowVector3d blue(149.0/255, 217.0/255, 244.0/255);
    //     const Eigen::RowVector3d red(250.0/255, 114.0/255, 104.0/255);
    //     const Eigen::RowVector3d gray(200.0/255, 200.0/255, 200.0/255);

    //     if(state.polyhedral)
    //     {
    //         Eigen::MatrixXd Vframe = (1.0 - anim_t) * (state.place_constraints? V: U) + anim_t * U1;
            
    //         viewer.data().set_colors(blue);
    //         viewer.data().set_vertices(Vframe);
    //         viewer.data().compute_normals();
            
    //         if(anim_t < 1.0f) anim_t += 0.01f;
    //     }
    //     else if(state.place_constraints)
    //     {
    //         viewer.data().clear();
    //         viewer.data().face_based = true;
    //         viewer.data().set_mesh(V,F);
    //         viewer.data().set_colors(gray);
    //         viewer.data().set_points(state.CV, red);

    //         // draw bounding box
    //         MatrixXd V_box;
    //         MatrixXi E_box;
    //         get_bounding_box(V, V_box, E_box);
    //         viewer.data().add_points(V_box, red);
    //         for (unsigned i=0;i<E_box.rows(); ++i)
    //             viewer.data().add_edges(V_box.row(E_box(i,0)),V_box.row(E_box(i,1)),red);
    //     }
    //     else
    //     {
    //         cube_style_single_iteration(V,U,data);
    //         viewer.data().clear();
    //         viewer.data().face_based = true;
    //         viewer.data().set_mesh(U,F);
    //         viewer.data().set_colors(blue);
    //         viewer.data().set_points(state.CV, red);

    //         // draw bounding box
    //         MatrixXd V_box;
    //         MatrixXi E_box;
    //         get_bounding_box(U, V_box, E_box);
    //         viewer.data().add_points(V_box, red);
    //         for (unsigned i=0;i<E_box.rows(); ++i)
    //             viewer.data().add_edges(V_box.row(E_box(i,0)),V_box.row(E_box(i,1)),red);
    //     }
    // };

    // when key pressed do 
    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod)
    {
        switch(key)
        {
            case '>':
            case '.':
            {
                data.lambda += 0.02;
                break;
            }
            case '<':
            case ',':
            {
                data.lambda -= 0.02;
                break;
            }
            case 'R':
            case 'r':
            {
                if (state.place_constraints)
                {
                    state.CV = MatrixXd();
                    state.CV_row_col = MatrixXi();
                    state.CV_row_col.resize(maxCV,2);
                    state.numCV = 0;
                    V = VO;
                }
                break;
            }
            case 'Q':
            case 'q':
            {
                U = (Rx * U.transpose()).transpose();
                if (state.place_constraints)
                {
                    V = (Rx * V.transpose()).transpose();
                    resetCV();
                    draw();
                }
                break;
            }
            case 'W':
            case 'w':
            {
                U = (Rx.transpose() * U.transpose()).transpose();
                if (state.place_constraints)
                {
                    V = (Rx.transpose() * V.transpose()).transpose();
                    resetCV();
                    draw();
                }
                break;
            }
            case 'A':
            case 'a':
            {
                U = (Ry * U.transpose()).transpose();
                if (state.place_constraints)
                {
                    V = (Rz * V.transpose()).transpose();
                    resetCV();
                    draw();
                }
                break;
            }
            case 'S':
            case 's':
            {
                U = (Ry.transpose() * U.transpose()).transpose();
                if (state.place_constraints)
                {
                    V = (Rx.transpose() * V.transpose()).transpose();
                    resetCV();
                    draw();
                }
                break;
            }
            case 'Z':
            case 'z':
            {
                U = (Rz * U.transpose()).transpose();
                if (state.place_constraints)
                {
                    V = (Rz * V.transpose()).transpose();
                    resetCV();
                    draw();
                }
                break;
            }
            case 'X':
            case 'x':
            {
                U = (Rz.transpose() * U.transpose()).transpose();
                if (state.place_constraints)
                {
                    V = (Rz.transpose() * V.transpose()).transpose();
                    resetCV();
                    draw();
                }
                break;
            }
            case ' ':
            {
                state.place_constraints = !state.place_constraints; // switch mode
                // reset poly_mode
                anim_t = 0.0;
                state.polyhedral = false;
                if (state.place_constraints)
                {
                    resetCV();
                    draw();
                }
                if (!state.place_constraints && state.CV.rows()>0)
                {
                    // set constrained points
                    igl::snap_points(state.CV, V, data.b);
                    data.bc.setZero(data.b.size(), 3);
                    for (int ii=0; ii<data.b.size(); ii++)
                        data.bc.row(ii) = V.row(data.b(ii));

                    // pre computation
                    U = V;
                    // cube_style_precomputation(V,F,data);
                    if(F.cols() == 4 && useQuadMesh)
                        cube_style_precomputation_quad(V,F,data);
                    else
                        cube_style_precomputation(V,F,data);
                    }
                else if(!state.place_constraints && state.CV.rows()==0) 
                {
                    // if not constraint points, then set the F(0,0) to be the contrained point
                    state.CV = MatrixXd();
                    state.CV.resize(1,3);
                    RowVector3d new_c = V.row(F(0,0));
                    state.CV.row(0) << new_c;
                    state.CV_row_col = MatrixXi();
                    state.CV_row_col.resize(maxCV,2);
                    state.CV_row_col.row(0) << 0, 0;
                    state.numCV = 1;

                    // set constrained points
                    igl::snap_points(state.CV, V, data.b);
                    data.bc.setZero(data.b.size(), 3);
                    for (int ii=0; ii<data.b.size(); ii++)
                        data.bc.row(ii) = V.row(data.b(ii));

                    // pre computation
                    U = V;
                    // cube_style_precomputation(V,F,data);
                    if(F.cols() == 4 && useQuadMesh)
                        cube_style_precomputation_quad(V,F,data);
                    else
                        cube_style_precomputation(V,F,data);
                    }
                break;
            }
            case 'B':
            case 'b':
            {
                state.polyhedral = !state.polyhedral;
                draw();
            }
            default:
                return false;
        }
        //draw();
        return true;
    };

    // when click mouse
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer&, int, int)->bool
    {
        if(state.place_constraints)
        {
            // Find closest point on mesh to mouse position
            double x = viewer.current_mouse_x;
            double y = viewer.core().viewport(3) - viewer.current_mouse_y;
            int fid;
            Vector3f bary;
            if(igl::unproject_onto_mesh(Vector2f(x,y), viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, fid, bary))
            {
                long c;
                bary.maxCoeff(&c);
                
                state.CV_row_col.row(state.numCV) << fid, c;
                state.numCV++;

                RowVector3d new_c = V.row(F(fid,c));
                if(state.CV.size()==0 || (state.CV.rowwise()-new_c).rowwise().norm().minCoeff() > 0)
                {
                    state.CV.conservativeResize(state.CV.rows()+1,3);
                    state.CV.row(state.CV.rows()-1) = new_c;
                    draw();
                    return true;
                }
            }
        }
        return false;
    };

    // default mode: keep drawing the current mesh
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &)->bool
    {
        if(viewer.core().is_animating && (!state.place_constraints || state.polyhedral))
            draw();
        return false;
    };

    // initialize the scene
    {   
        viewer.data().set_mesh(V,F);
        viewer.data().show_lines=false;
        viewer.core().is_animating=true;
        viewer.data().face_based=true;
        Vector4f backColor;
        backColor << bg_color(0), bg_color(1), bg_color(2), 1.f;
        viewer.core().background_color = backColor;
        load_mesh(current_model);
        draw();
        viewer.launch();
    } 

}