#include <cinolib/gl/glcanvas.h>
#include <cinolib/meshes/drawable_hexmesh.h>
#include <cinolib/meshes/polygonmesh.h>
#include <cinolib/export_visible.h>
#include <cinolib/gl/volume_mesh_controls.h>
#include <cinolib/gl/file_dialog_open.h>
#include <cinolib/geometry/plane.h>
#include <cinolib/geometry/triangle_utils.h>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <imgui.h>
#include <svg.hpp>
#include <optional>
#include <algorithm>
#include <utility>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/intersections.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Barycentric_coordinates_2/triangle_coordinates_2.h>
#include <CGAL/centroid.h>
#include <unordered_set>
#include <set>
#include <queue>

struct Poly final
{

	std::array<cinolib::vec3d, 3> projTriVerts[2];
	std::vector<SVG::Point> points;
	double lambert;
	bool onSurf;
	double z;
};

std::vector<Poly> getPolys(const cinolib::DrawableHexmesh<>& _mesh, const cinolib::GLcanvas& _canvas)
{
	std::vector<Poly> polys;
	polys.reserve(_mesh.num_faces() / 2);
	const double w{ static_cast<double>(_canvas.canvas_width()) }, h{ static_cast<double>(_canvas.height()) };
	const cinolib::mat4d& mat{ _canvas.camera.projectionViewMatrix() };
	std::vector<cinolib::vec3d> projVerts(_mesh.vector_verts().size());
	for (std::size_t i{}; i < projVerts.size(); i++)
	{
		projVerts[i] = ((mat * _mesh.vector_verts()[i]) + cinolib::vec3d{ 1.0 }) / 2.0;
		projVerts[i].x() *= w;
		projVerts[i].y() *= -h;
		projVerts[i].y() += h;
	}
	const cinolib::vec3d& cameraBack{ _canvas.camera.view.normBack() };
	for (unsigned int fid{}; fid < _mesh.num_faces(); fid++)
	{
		unsigned int pid;
		if (_mesh.face_is_visible(fid, pid))
		{
			Poly poly;
			const bool cw{ _mesh.poly_face_is_CW(pid, fid) };
			bool anyFront{ false };
			cinolib::vec3d normals{};
			for (std::size_t ti{}; ti < 2; ti++)
			{
				std::array<unsigned int, 3> triVids{
					_mesh.face_tessellation(fid)[ti * 3 + (cw ? 2 : 0)],
					_mesh.face_tessellation(fid)[ti * 3 + 1],
					_mesh.face_tessellation(fid)[ti * 3 + (cw ? 0 : 2)]
				};
				for (std::size_t vi{}; vi < 3; vi++)
				{
					poly.projTriVerts[ti][vi] = projVerts[static_cast<std::size_t>(triVids[vi])];
				}
				const cinolib::vec3d normal{ cinolib::triangle_normal(
					_mesh.vert(triVids[0]),
					_mesh.vert(triVids[1]),
					_mesh.vert(triVids[2])
				) };
				if (anyFront || normal.dot(cameraBack) > 0.0)
				{
					anyFront = true;
				}
				normals += normal;
			}
			const cinolib::vec3d normal{ (normals / 2.0).normalized() };
			poly.lambert = normal.dot(cameraBack);
			if (poly.lambert > 0.0)
			{
				anyFront = true;
			}
			else
			{
				poly.lambert = 0.0;
			}
			if (!anyFront)
			{
				continue;
			}
			poly.lambert *= (_mesh.face_data(fid).AO * _mesh.AO_alpha) + (1.0 - _mesh.AO_alpha);
			poly.z = 0.0;
			poly.points.reserve(4);
			for (const unsigned int vid : _mesh.adj_f2v(fid))
			{
				const cinolib::vec3d& projVert{ projVerts[static_cast<std::size_t>(vid)] };
				poly.z = projVert.z();
				poly.points.emplace_back(
					projVert.x(),
					projVert.y()
				);
			}
			poly.z /= 4.0;
			poly.onSurf = _mesh.face_is_on_srf(fid);
			polys.push_back(std::move(poly));
		}
	}
	return polys;
}

enum EPolyCompareResult
{
	Behind, Above, Disjoint
};

#ifdef DEBUG_PRINT_INTERSECTIONS
std::vector<std::vector<SVG::Point>> debugIntersections{};
std::vector<SVG::Point> debugSamples{};
#endif

EPolyCompareResult comparePolys(const Poly& _a, const Poly& _b)
{
	using Kernel = CGAL::Simple_cartesian<double>;
	using Point = Kernel::Point_2;
	using Segment = Kernel::Segment_2;
	using Triangle = Kernel::Triangle_2;
	std::vector<Kernel::FT> coords;
	coords.reserve(3);
	for (std::size_t ati{}; ati < 2; ati++)
	{
		for (std::size_t bti{}; bti <= ati; bti++)
		{
			const Triangle a{
				Point{ _a.projTriVerts[ati][0].x(), _a.projTriVerts[ati][0].y() },
				Point{ _a.projTriVerts[ati][1].x(), _a.projTriVerts[ati][1].y() },
				Point{ _a.projTriVerts[ati][2].x(), _a.projTriVerts[ati][2].y() }
			};
			const Triangle b{
				Point{ _b.projTriVerts[bti][0].x(), _b.projTriVerts[bti][0].y() },
				Point{ _b.projTriVerts[bti][1].x(), _b.projTriVerts[bti][1].y() },
				Point{ _b.projTriVerts[bti][2].x(), _b.projTriVerts[bti][2].y() }
			};
			if (a.is_degenerate() || b.is_degenerate())
			{
				continue;
			}
			decltype(CGAL::intersection(a, b)) intersOrNone{};
			try
			{
				intersOrNone = CGAL::intersection(a, b);
			}
			catch (...)
			{
				continue;
			}
			if (intersOrNone)
			{
				const auto decide{ [&]()
				{
					const auto compareAt{ [&](const std::vector<Point>& _points)
					{
						const Point center{ CGAL::centroid(_points.begin(), _points.end())};
#ifdef DEBUG_PRINT_INTERSECTIONS
						{
							std::vector<SVG::Point> dps{};
							for (const auto& p : _points)
							{
								dps.emplace_back(p.x(), p.y());
							}
							debugIntersections.push_back(dps);
							debugSamples.emplace_back(center.x(), center.y());
						}
#endif
						const auto sample{ [&](const std::array<cinolib::vec3d, 3>& _p)
						{
							coords.clear();
							CGAL::Barycentric_coordinates::triangle_coordinates_2(
								Point{ _p[0].x(), _p[0].y() },
								Point{ _p[1].x(), _p[1].y() },
								Point{ _p[2].x(), _p[2].y() },
								center,
								std::back_inserter(coords)
							);
							return
								_p[0].z() * coords[0] +
								_p[1].z() * coords[1] +
								_p[2].z() * coords[2];
						} };
						const double diff{ sample(_a.projTriVerts[ati]) - sample(_b.projTriVerts[bti]) };
						if (std::abs(diff) < 1e-20)
						{
							return EPolyCompareResult::Disjoint;
						}
						return diff > 0 ? EPolyCompareResult::Behind : EPolyCompareResult::Above;
					}
};
const auto inters{ *intersOrNone };
switch (inters.which())
{
	case 0: // point
	{
		//const Point& ab{ boost::get<Point>(inters) };
		//return decide({ ab });
		break;
	}
	case 1: // segment
	{
		//const Segment& ab{ boost::get<Segment>(inters) };
		//return decide({ ab[0], ab[1] });
		break;
	}
	case 2: // triangle
	{
		const Triangle& ab{ boost::get<Triangle>(inters) };
		return compareAt({ ab[0], ab[1], ab[2] });
	}
	case 3: // points
	{
		const std::vector<Point>& ab{ boost::get<std::vector<Point>>(inters) };
		return compareAt(ab);
	}
}
return EPolyCompareResult::Disjoint;
}
				};
				const EPolyCompareResult result{ decide() };
				if (result != EPolyCompareResult::Disjoint)
				{
					return result;
				}
			}
		}
	}
	return EPolyCompareResult::Disjoint;
}

std::vector<Poly> sortPolys(const std::vector<Poly>& _polys)
{
	struct Node final
	{
		std::size_t beforeCount{};
		std::vector<std::size_t> afterThis{};
	};
	std::size_t cmps{}, inters{};
	std::vector<Node> nodes(_polys.size(), Node{});
	for (std::size_t a{}; a < _polys.size(); a++)
	{
		for (std::size_t b{}; b < a; b++)
		{
			cmps++;
			const EPolyCompareResult cmp{ comparePolys(_polys[a], _polys[b]) };
			switch (cmp)
			{
				case EPolyCompareResult::Above:
					nodes[a].beforeCount++;
					nodes[b].afterThis.push_back(a);
					inters++;
					break;
				case EPolyCompareResult::Behind:
					nodes[a].afterThis.push_back(b);
					nodes[b].beforeCount++;
					inters++;
					break;
				default:
					break;
			}
		}
	}
	std::set<std::size_t> nonFree;
	std::queue<std::size_t> free;
	for (std::size_t i{}; i < nodes.size(); i++)
	{
		if (nodes[i].beforeCount == 0)
		{
			free.push(i);
		}
		else
		{
			nonFree.insert(i);
		}
	}
	std::vector<Poly> sortedPolys;
	sortedPolys.reserve(_polys.size());
	do
	{
		while (!free.empty())
		{
			const std::size_t i{ free.front() };
			free.pop();
			sortedPolys.push_back(_polys[i]);
			for (const std::size_t after : nodes[i].afterThis)
			{
				nodes[after].beforeCount--;
				if (nodes[after].beforeCount == 0)
				{
					nonFree.erase(after);
					free.push(after);
				}
			}
		}
		if (!nonFree.empty())
		{
			const auto it{ nonFree.begin() };
			nodes[*it].beforeCount = 0;
			free.push(*it);
			nonFree.erase(it);
		}
	}
	while (!free.empty() && !nonFree.empty());
	std::cout << _polys.size() << " polys, " << cmps << " comparations, " << inters << " intersections, " << sortedPolys.size() << " sorted polys" << std::endl;
	return sortedPolys;
}

void sortPolysByZ(std::vector<Poly>& _polys)
{
	std::sort(_polys.begin(), _polys.end(), [](const Poly& _a, const Poly& _b) { return _a.z > _b.z; });
}

std::string render(const cinolib::DrawableHexmesh<>& _mesh, const cinolib::GLcanvas& _canvas)
{
	const double w{ static_cast<double>(_canvas.canvas_width()) }, h{ static_cast<double>(_canvas.height()) };
	SVG::SVG svg;
	svg.set_attr("width", w).set_attr("height", h);
	SVG::Group& group{ *svg.add_child<SVG::Group>() };
	group.set_attr("stroke", "black").set_attr("stroke-width", "0.75").set_attr("stroke-linecap", "round").set_attr("stroke-linejoin", "round");

#ifdef DEBUG_PRINT_INTERSECTIONS
	debugIntersections.clear();
	debugSamples.clear();
#endif

	std::vector<Poly> polys{ getPolys(_mesh, _canvas) };
	sortPolysByZ(polys);
	polys = sortPolys(polys);

	for (const Poly& poly : polys)
	{
		SVG::Polygon& polygon{ *group.add_child<SVG::Polygon>(poly.points) };
		{
			const double value{ poly.onSurf
				? 0.55 * (1.0 - poly.lambert) + 0.9 * poly.lambert
				: 0.65 * (1.0 - poly.lambert) + 1.0 * poly.lambert
			};
			const double saturation{ poly.onSurf
				? 0.0
				: 1.0 * (1 - poly.lambert) + 0.7 * poly.lambert
			};
			cinolib::Color color{ cinolib::Color::hsv2rgb(
				32.0f / 360.0f,
				static_cast<float>(saturation),
				static_cast<float>(value)
			) };
			std::ostringstream ss{};
			ss << "rgb"
				<< '('
				<< static_cast<int>(color.r() * 255)
				<< ','
				<< static_cast<int>(color.g() * 255)
				<< ','
				<< static_cast<int>(color.b() * 255)
				<< ')';
			polygon.set_attr("fill", ss.str());
		}
	}

#ifdef DEBUG_PRINT_INTERSECTIONS
	for (std::size_t i{}; i < debugIntersections.size(); i++)
	{
		cinolib::Color color{ cinolib::Color::hsv2rgb(
				static_cast<float>(i) / static_cast<float>(debugIntersections.size()),
				1.0f,
				1.0f
			) };
		std::ostringstream ss{};
		ss << "rgb"
			<< '('
			<< static_cast<int>(color.r() * 255)
			<< ','
			<< static_cast<int>(color.g() * 255)
			<< ','
			<< static_cast<int>(color.b() * 255)
			<< ')';
		SVG::Polygon& polygon{ *group.add_child<SVG::Polygon>(debugIntersections[i]) };
		polygon.set_attr("stroke", ss.str());
		group.add_child<SVG::Circle>(debugSamples[i].first, debugSamples[i].second, 1)->set_attr("fill", ss.str()).set_attr("stroke", ss.str());
	}
#endif

	return std::string{ svg };
}


int main(int _argc, char* _argv[])
{
	std::optional<std::string> filename{ std::nullopt };
	if (_argc == 2)
	{
		filename = _argv[1];
	}
	else if (_argc > 2)
	{
		std::cerr << "expected 0 or 1 argument, got " << _argc - 1 << std::endl;
		return 1;
	}
	if (!filename)
	{
		std::string filenameOrEmpty{ cinolib::file_dialog_open() };
		if (!filenameOrEmpty.empty())
		{
			filename = filenameOrEmpty;
		}
	}
	if (!filename)
	{
		std::cerr << "no file opened" << std::endl;
		return 2;
	}
	if (!std::filesystem::exists(*filename))
	{
		std::cout << "file '" << *filename << "' does not exist" << std::endl;
		return 3;
	}
	if (std::filesystem::path{ *filename }.extension() != ".mesh")
	{
		std::cout << "file '" << *filename << "' is not a .mesh file" << std::endl;
		return 4;
	}
	std::cout << "loading file '" << *filename << "'" << std::endl;
	cinolib::DrawableHexmesh<> mesh{ filename->c_str() };
	cinolib::GLcanvas canvas{ 900, 800, 13, 1 };
	cinolib::VolumeMeshControls<cinolib::DrawableHexmesh<>> controls{ &mesh, &canvas };
	canvas.push(&controls);
	canvas.push(&mesh);
	canvas.background = cinolib::Color::hsv2rgb(.0f, .0f, .09f);
	canvas.callback_app_controls = [&]()
	{
		if (ImGui::Button("Export"))
		{
			const std::string outFilename{ cinolib::file_dialog_save() };
			if (!outFilename.empty())
			{
				std::ofstream file;
				file.open(outFilename);
				file << render(mesh, canvas);
				file.close();
				std::cout << "written file '" << outFilename << "'" << std::endl;
			}
		}
	};
	mesh.poly_set_color(cinolib::Color::hsv2rgb(.0f, .0f, .25f));
	mesh.edge_set_color(cinolib::Color::hsv2rgb(.0f, .0f, .0f));
	mesh.show_mesh_flat();
	mesh.show_in_wireframe_width(2.f);
	mesh.show_out_wireframe_width(2.f);
	std::cout << mesh.genus() << std::endl;
	return canvas.launch();
}
