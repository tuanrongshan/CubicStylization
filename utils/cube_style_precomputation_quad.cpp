#include "cube_style_precomputation_quad.h"

#include <igl/slice.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/arap_rhs.h>
#include <igl/vertex_triangle_adjacency.h>

#include <Eigen/Sparse>
#include <vector>
#include <array>
#include <stdexcept>
#include <unordered_set>

static inline Eigen::RowVector3d face_normal(
    const Eigen::MatrixXd &V,
    const Eigen::RowVector4i &f)
{
    Eigen::Vector3d p0 = V.row(f(0)).transpose();
    Eigen::Vector3d p1 = V.row(f(1)).transpose();
    Eigen::Vector3d p2 = V.row(f(2)).transpose();
    Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
    double len = n.norm();
    if (len > 0.0)
        n /= len;
    return n.transpose();
}

static void quad_vertex_normals(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::MatrixXd &N)
{
    int nv = V.rows();
    N.setZero(nv, 3);
    for (int f = 0; f < F.rows(); ++f)
    {
        auto fn = face_normal(V, F.row(f));
        for (int k = 0; k < 4; ++k)
            N.row(F(f, k)) += fn;
    }
    for (int v = 0; v < nv; ++v)
    {
        double len = N.row(v).norm();
        if (len > 0.0)
            N.row(v) /= len;
    }
}

static Eigen::VectorXd quad_vertex_areas(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F)
{
    int nv = V.rows();
    Eigen::VectorXd A = Eigen::VectorXd::Zero(nv);
    for (int f = 0; f < F.rows(); ++f)
    {
        Eigen::Vector3d p0 = V.row(F(f, 0)).transpose();
        for (int k = 1; k < 3; ++k) // triangles (0,1,2) and (0,2,3)
        {
            int i1 = F(f, k), i2 = F(f, k + 1);
            Eigen::Vector3d p1 = V.row(i1).transpose();
            Eigen::Vector3d p2 = V.row(i2).transpose();
            double area = 0.5 * ((p1 - p0).cross(p2 - p0)).norm() / 3.0;
            A(F(f, 0)) += area;
            A(i1) += area;
            A(i2) += area;
        }
    }
    return A;
}

static inline uint64_t edge_key(int a, int b)
{
    return (a < b) ? ((uint64_t)a << 32 | (uint32_t)b)
                   : ((uint64_t)b << 32 | (uint32_t)a);
}
static void quad_edges(
    const Eigen::MatrixXi &F,
    std::vector<std::array<int, 2>> &E)
{
    struct Hash
    {
        size_t operator()(uint64_t x) const noexcept { return (size_t)x; }
    };
    std::unordered_set<uint64_t, Hash> S;
    for (int f = 0; f < F.rows(); ++f)
        for (int k = 0; k < 4; ++k)
        {
            int i = F(f, k), j = F(f, (k + 1) & 3);
            uint64_t key = edge_key(i, j);
            if (S.emplace(key).second)
                E.push_back({{i, j}});
        }
}

void cube_style_precomputation_quad(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    cube_style_data &data)
{
    if (F.cols() != 4)
        throw std::runtime_error("cube_style_precomputation_quad expects an n×4 face list.");

    data.reset();
    int nV = V.rows();

    // 1) build per-vertex normals
    quad_vertex_normals(V, F, data.N);

    // 2) temporary triangulation *only for cotans & mass*
    Eigen::MatrixXi Ftri(F.rows() * 2, 3);
    for (int f = 0; f < F.rows(); ++f)
    {
        auto q = F.row(f);
        Ftri.row(2 * f) << q(0), q(1), q(2);
        Ftri.row(2 * f + 1) << q(0), q(2), q(3);
    }

    // 3) cotangent Laplacian & mass matrix
    igl::cotmatrix(V, Ftri, data.L);
    {
        Eigen::SparseMatrix<double> M;
        igl::massmatrix(
            V, Ftri,
            igl::MASSMATRIX_TYPE_BARYCENTRIC,
            M);
        data.VA = M.diagonal();
    }

    // 4) one‐ring on the *tri* mesh (for ARAP rhs only)
    std::vector<std::vector<int>> adjF, VI;
    igl::vertex_triangle_adjacency(nV, Ftri, adjF, VI);

    // 5) ARAP rhs via libigl
    igl::arap_rhs(
        V, Ftri, V.cols(),
        igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS,
        data.K);

    // 6) build quad half‐edge lists & dV with **cotan weights**
    // Build per-vertex neighbor lists from Ftri
    std::vector<std::vector<int>> nbr(nV);
    for (int t = 0; t < Ftri.rows(); ++t)
    {
        int a = Ftri(t, 0), b = Ftri(t, 1), c = Ftri(t, 2);
        nbr[a].push_back(b);
        nbr[a].push_back(c);
        nbr[b].push_back(a);
        nbr[b].push_back(c);
        nbr[c].push_back(a);
        nbr[c].push_back(b);
    }
    // uniquify
    for (auto &vlist : nbr)
    {
        std::sort(vlist.begin(), vlist.end());
        vlist.erase(std::unique(vlist.begin(), vlist.end()), vlist.end());
    }

    // now fill your data structures exactly as in the triangle version:
    data.hEList.resize(nV);
    data.WVecList.resize(nV);
    data.dVList.resize(nV);

    for (int v = 0; v < nV; ++v)
    {
        const auto &N = nbr[v];
        int m = (int)N.size(); // number of spokes

        data.hEList[v].resize(m, 2);
        data.WVecList[v].resize(m);
        data.dVList[v].resize(3, m);

        // spokes = (v → neighbor) with the cotan weight from data.L
        for (int k = 0; k < m; ++k)
        {
            int w = N[k];
            data.hEList[v](k, 0) = v;
            data.hEList[v](k, 1) = w;
            // non-zero cotan weight, because (v,w) is in some triangle of Ftri
            data.WVecList[v](k) = data.L.coeff(v, w);
        }

        // build dV exactly like the triangle code
        Eigen::MatrixXd Va, Vb;
        igl::slice(V, data.hEList[v].col(0), 1, Va);
        igl::slice(V, data.hEList[v].col(1), 1, Vb);
        data.dVList[v] = (Vb - Va).transpose(); // 3×m
    }

    igl::min_quad_with_fixed_precompute(
        data.L, data.b,
        Eigen::SparseMatrix<double>(),
        false, data.solver_data);

    data.zAll.resize(3, V.rows()); data.zAll.setRandom();
    data.uAll.resize(3, V.rows()); data.uAll.setRandom();
    data.rhoAll.resize(V.rows()); data.rhoAll.setConstant(data.rhoInit);
}
