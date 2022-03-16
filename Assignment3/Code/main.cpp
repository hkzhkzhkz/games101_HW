#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"
#include <math.h>

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    //  Use the same projection matrix from the previous assignments
	Eigen::Matrix4f projection;

	Eigen::Matrix4f ortho = Eigen::Matrix4f::Identity();
	float angle = (eye_fov / 2) / 180 * acos(-1);
	Eigen::Matrix4f ortho1 = Eigen::Matrix4f::Identity();
	ortho1 << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, -(zNear + zFar) / 2,
		0, 0, 0, 1;
	Eigen::Matrix4f ortho2 = Eigen::Matrix4f::Identity();
	ortho2 << 1 / (tan(angle) * zNear * aspect_ratio), 0, 0, 0,
		0, 1 / (tan(angle) * zNear), 0, 0,
		0, 0, 2 / (zNear - zFar), 0,
		0, 0, 0, 1;
	ortho = ortho2 * ortho1;

	Eigen::Matrix4f persp = Eigen::Matrix4f::Identity();
	persp << zNear, 0, 0, 0,
		0, zNear, 0, 0,
		0, 0, zNear + zFar, -zNear * zFar,
		0, 0, 1, 0;

	projection = ortho * persp;

	return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position; ///光源位置
    Eigen::Vector3f intensity;///光强度
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);    //全局环境光照系数
    Eigen::Vector3f kd = texture_color / 255.f;                   //漫反射光照系数
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937); //高光系数

    auto l1 = light{{20, 20, 20}, {500, 500, 500}}; ///光源位置以及光源强度
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};///全局环境光照强度
    Eigen::Vector3f eye_pos{0, 0, 10}; ///人眼观测位置

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
		auto ambient = Eigen::Vector3f(ka.x() * amb_light_intensity.x(), ka.y() * amb_light_intensity.y(), ka.z() * amb_light_intensity.z());
		auto l = (light.position - point).normalized();///由被观察点指向光源方向
		auto v = (eye_pos - point).normalized();
		auto h = (v + l).normalized();///半程向量
		auto r = (light.position - point).norm();///被观察点到光源的距离
		auto tmp1 = light.intensity / (r * r);/// I/r*r
		auto diffuse = Eigen::Vector3f(kd.x() * tmp1.x(), kd.y() * tmp1.y(), kd.z() * tmp1.z()) * std::max(0.0f, normal.dot(l));
		auto specular = Eigen::Vector3f(ks.x() * tmp1.x(), ks.y() * tmp1.y(), ks.z() * tmp1.z()) * pow(std::max(0.0f, normal.dot(h)), p);
        
		result_color += (ambient + diffuse + specular);
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        
		auto ambient = Eigen::Vector3f(ka.x() * amb_light_intensity.x(), ka.y() * amb_light_intensity.y(), ka.z() * amb_light_intensity.z());
		auto l = (light.position - point).normalized();///由被观察点指向光源方向
		auto v = (eye_pos - point).normalized();
		auto h = (v + l).normalized();///半程向量
		auto r = (light.position - point).norm();///被观察点到光源的距离
		auto tmp1 = light.intensity / (r * r);/// I/r*r
		auto diffuse = Eigen::Vector3f(kd.x() * tmp1.x(), kd.y() * tmp1.y(), kd.z() * tmp1.z()) * std::max(0.0f, normal.dot(l));
		auto specular = Eigen::Vector3f(ks.x() * tmp1.x(), ks.y() * tmp1.y(), ks.z() * tmp1.z()) * pow(std::max(0.0f, normal.dot(h)), p);

		result_color += (ambient + diffuse + specular);

    }

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
	Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
	Eigen::Vector3f kd = payload.color;
	Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

	auto l1 = light{ {20, 20, 20}, {500, 500, 500} };
	auto l2 = light{ {-20, 20, 0}, {500, 500, 500} };

	std::vector<light> lights = { l1, l2 };
	Eigen::Vector3f amb_light_intensity{ 10, 10, 10 };
	Eigen::Vector3f eye_pos{ 0, 0, 10 };

	float p = 150;

	Eigen::Vector3f color = payload.color;
	Eigen::Vector3f point = payload.view_pos;
	Eigen::Vector3f normal = payload.normal;


	float kh = 0.2, kn = 0.1;

	// TODO: Implement bump mapping here
	auto x = payload.normal.x();
	auto y = payload.normal.y();
	auto z = payload.normal.z();
	// Let n = normal = (x, y, z)
	auto n = payload.normal;
	// Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
	Eigen::Vector3f t(x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
	// Vector b = n cross product t
	Eigen::Vector3f b = n.cross(t);
	// Matrix TBN = [t b n]
	Eigen::Matrix<float, 3, 3> TBN;
	TBN << t[0], b[0], n[0]
		, t[1], b[1], n[1]
		, t[2], b[2], n[2];
	auto u = payload.tex_coords[0];
	auto v = payload.tex_coords[1];
	auto w = payload.texture->width;
	auto h = payload.texture->height;
	// dU = kh * kn * (h(u+1/w,v)-h(u,v))
	float hu1 = payload.texture->getColor(u + 1.0 / w, v).norm();
	float hu0 = payload.texture->getColor(u, v).norm();
	auto dU = kh * kn * (hu1 - hu0);
	// dV = kh * kn * (h(u,v+1/h)-h(u,v))
	float hv1 = payload.texture->getColor(u, v + 1.0 / h).norm();
	float hv0 = hu0;
	float dV = kh * kn * (hv1 - hv0);
	// Vector ln = (-dU, -dV, 1)
	Eigen::Vector3f ln(-dU, -dV, 1);
	// Normal n = normalize(TBN * ln)
	normal = TBN * ln;
	normal = normal.normalized();///只改变法向量

	Eigen::Vector3f result_color = { 0, 0, 0 };
	result_color = normal;

	return result_color * 255.f;
}

Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)


	auto x = payload.normal.x();
	auto y = payload.normal.y();
	auto z = payload.normal.z();
	auto n = payload.normal;
	Eigen::Vector3f t(x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
	Eigen::Vector3f b = n.cross(t);
	Eigen::Matrix<float, 3, 3> TBN;
	TBN << t[0], b[0], n[0]
		, t[1], b[1], n[1]
		, t[2], b[2], n[2];
	auto u = payload.tex_coords[0];
	auto v = payload.tex_coords[1];
	auto w = payload.texture->width;
	auto h = payload.texture->height;
	float hu1 = payload.texture->getColor(u + 1.0 / w, v).norm();
	float hu0 = payload.texture->getColor(u, v).norm();
	auto dU = kh * kn * (hu1 - hu0);
	float hv1 = payload.texture->getColor(u, v + 1.0 / h).norm();
	float hv0 = hu0;
	float dV = kh * kn * (hv1 - hv0);
	Eigen::Vector3f ln(-dU, -dV, 1);

	point += (kn * normal * payload.texture->getColor(u, v).norm());///改变点的真实位置
	normal = TBN * ln;
	normal = normal.normalized();


    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.

		//auto ambient = Eigen::Vector3f(ka.x() * amb_light_intensity.x(), ka.y() * amb_light_intensity.y(), ka.z() * amb_light_intensity.z());
		auto ambient = ka.cwiseProduct(amb_light_intensity);
        auto l = (light.position - point).normalized();///由被观察点指向光源方向
		auto v = (eye_pos - point).normalized();
		auto h = (v + l).normalized();///半程向量
		auto r = (light.position - point).norm();///被观察点到光源的距离
		auto tmp1 = light.intensity / (r * r);/// I/r*r
		//auto diffuse = Eigen::Vector3f(kd.x() * tmp1.x(), kd.y() * tmp1.y(), kd.z() * tmp1.z()) * std::max(0.0f, normal.dot(l));
        auto diffuse = kd.cwiseProduct(tmp1) * std::max(0.0f, normal.dot(l));
		//auto specular = Eigen::Vector3f(ks.x() * tmp1.x(), ks.y() * tmp1.y(), ks.z() * tmp1.z()) * pow(std::max(0.0f, normal.dot(h)), p);
        auto specular = ks.cwiseProduct(tmp1) * pow(std::max(0.0f, normal.dot(h)), p);

		result_color += (ambient + diffuse + specular);
    }

    return result_color * 255.f;
}


int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "D:\\CG\\GAMES101\\homework\\Assignment3\\Code\\models\\spot\\";

    // Load .obj File
    bool loadout = Loader.LoadFile("D:\\CG\\GAMES101\\homework\\Assignment3\\Code\\models\\spot\\spot_triangulated_good.obj");

    //Set all triangle point attribution of obj to class Triangle() which ptr stored in TriangleList
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
				t->setVertex(j, Vector4f(mesh.Vertices[(double)i + j].Position.X, mesh.Vertices[(double)i + j].Position.Y, mesh.Vertices[(double)i + j].Position.Z, 1.0));
				t->setNormal(j, Vector3f(mesh.Vertices[(double)i + j].Normal.X, mesh.Vertices[(double)i + j].Normal.Y, mesh.Vertices[(double)i + j].Normal.Z));
				t->setTexCoord(j, Vector2f(mesh.Vertices[(double)i + j].TextureCoordinate.X, mesh.Vertices[(double)i + j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "spot_texture.png";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = texture_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);///设置顶点着色器
    r.set_fragment_shader(active_shader);///

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
