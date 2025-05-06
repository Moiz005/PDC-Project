#include <mpi.h>
#include <omp.h>
#include <metis.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <random>
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <set>
#include <functional>

using real = double;
const real INF = std::numeric_limits<real>::infinity();

// CSR on rank 0
struct CSR
{
    int n;
    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    std::vector<real> adjwgt;
};

// Read 4-column "idx u v w", ignore idx
CSR readCSR(const char *fname)
{
    std::ifstream fin(fname);
    if (!fin)
    {
        std::cerr << "Cannot open " << fname << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int idx, u, v;
    real w;
    std::vector<std::tuple<int, int, real>> edges;
    int maxv = -1;
    while (fin >> idx >> u >> v >> w)
    {
        edges.emplace_back(u, v, w);
        edges.emplace_back(v, u, w);
        maxv = std::max({maxv, u, v});
    }
    CSR G;
    G.n = maxv + 1;
    int M = edges.size();
    G.xadj.assign(G.n + 1, 0);
    for (auto &e : edges)
    {
        u = std::get<0>(e);
        G.xadj[u + 1]++;
    }
    for (int i = 1; i <= G.n; ++i)
        G.xadj[i] += G.xadj[i - 1];
    G.adjncy.resize(M);
    G.adjwgt.resize(M);
    std::vector<int> ptr(G.xadj.begin(), G.xadj.end());
    for (auto &e : edges)
    {
        std::tie(u, v, w) = e;
        int pos = ptr[u]++;
        G.adjncy[pos] = v;
        G.adjwgt[pos] = w;
    }
    return G;
}

// Partition via METIS
std::vector<int> metisPartition(const CSR &G, int np)
{
    idx_t nv = G.n, ncon = 1, nparts = np;
    std::vector<idx_t> xadj(G.xadj.begin(), G.xadj.end());
    std::vector<idx_t> adjncy(G.adjncy.begin(), G.adjncy.end());
    std::vector<idx_t> adjwgt(adjncy.size(), 1);
    std::vector<idx_t> part(nv), options(METIS_NOPTIONS);
    METIS_SetDefaultOptions(options.data());
    idx_t objval;
    int status = METIS_PartGraphKway(
        &nv, &ncon,
        xadj.data(), adjncy.data(),
        nullptr, nullptr,
        adjwgt.data(),
        &nparts,
        nullptr, nullptr,
        options.data(),
        &objval,
        part.data());
    if (status != METIS_OK)
    {
        std::cerr << "METIS failed\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return std::vector<int>(part.begin(), part.end());
}

// Local adjacency
struct Graph
{
    int n;
    std::vector<std::vector<std::pair<int, real>>> adj;
    Graph(int N = 0) : n(N), adj(N) {}
    void addEdge(int u, int v, real w)
    {
        adj[u].emplace_back(v, w);
    }
    void removeEdge(int u, int v)
    {
        auto &A = adj[u];
        A.erase(std::find_if(A.begin(), A.end(),
                             [&](auto &p)
                             { return p.first == v; }));
        auto &B = adj[v];
        B.erase(std::find_if(B.begin(), B.end(),
                             [&](auto &p)
                             { return p.first == u; }));
    }
};

// Dijkstra with OpenMP
void localDijkstra(const Graph &G, int src, std::vector<real> &dist)
{
    int N = G.n;
    dist.assign(N, INF);
    if (src >= 0 && src < N)
        dist[src] = 0.0;
    using P = std::pair<real, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    if (src >= 0 && src < N)
        pq.emplace(0.0, src);
    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u])
            continue;
        std::vector<P> to_push;
#pragma omp parallel
        {
            std::vector<P> buf;
#pragma omp for nowait
            for (int i = 0; i < (int)G.adj[u].size(); ++i)
            {
                auto [v, w] = G.adj[u][i];
                real nd = d + w;
                if (nd < dist[v])
                {
#pragma omp critical
                    {
                        if (nd < dist[v])
                        {
                            dist[v] = nd;
                            buf.emplace_back(nd, v);
                        }
                    }
                }
            }
#pragma omp critical
            to_push.insert(to_push.end(), buf.begin(), buf.end());
        }
        for (auto &p : to_push)
            pq.push(p);
    }
}

// Incremental update (Algs 2&3)
void ProcessCE(const Graph &G,
               const std::vector<std::tuple<int, int, real>> &del,
               const std::vector<std::tuple<int, int, real>> &ins,
               std::vector<real> &dist,
               std::vector<int> &parent,
               std::set<int> &AD,
               std::set<int> &A)
{
    AD.clear();
    A.clear();
    int N = G.n;
    for (auto &e : del)
    {
        int u, v;
        real w;
        std::tie(u, v, w) = e;
        if (parent[u] == v || parent[v] == u)
        {
            int y = (dist[u] > dist[v] ? u : v);
            dist[y] = INF;
            AD.insert(y);
            A.insert(y);
        }
    }
    for (auto &e : ins)
    {
        int u, v;
        real w;
        std::tie(u, v, w) = e;
        int x = (dist[u] > dist[v] ? v : u), y = (x == u ? v : u);
        if (dist[x] < INF && dist[x] + w < dist[y])
        {
            dist[y] = dist[x] + w;
            parent[y] = x;
            A.insert(y);
        }
    }
}

void UpdateAffectedVertices(const Graph &G,
                            std::vector<real> &dist,
                            std::vector<int> &parent,
                            std::set<int> &AD,
                            std::set<int> &A)
{
    int N = G.n;
    while (!AD.empty())
    {
        std::set<int> nextD;
        for (int v : AD)
            for (int c = 0; c < N; ++c)
                if (parent[c] == v)
                {
                    dist[c] = INF;
                    nextD.insert(c);
                    A.insert(c);
                }
        AD.swap(nextD);
    }
    while (!A.empty())
    {
        std::set<int> nextA;
        for (int v : A)
            for (auto &pr : G.adj[v])
            {
                int n = pr.first;
                real w = pr.second;
                if (dist[v] < INF && dist[v] + w < dist[n])
                {
                    dist[n] = dist[v] + w;
                    parent[n] = v;
                    nextA.insert(n);
                }
                if (dist[n] < INF && dist[n] + w < dist[v])
                {
                    dist[v] = dist[n] + w;
                    parent[v] = n;
                    nextA.insert(v);
                }
            }
        A.swap(nextA);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // --- Read & partition on rank 0 ---
    CSR G0;
    std::vector<int> part;
    if (rank == 0)
    {
        G0 = readCSR("graph.txt");
        part = metisPartition(G0, np);
    }

    // --- Broadcast CSR and partition arrays ---
    MPI_Bcast(&G0.n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank > 0)
        G0.xadj.resize(G0.n + 1);
    MPI_Bcast(G0.xadj.data(), G0.n + 1, MPI_INT, 0, MPI_COMM_WORLD);

    int M = G0.xadj.back();
    if (rank > 0)
    {
        G0.adjncy.resize(M);
        G0.adjwgt.resize(M);
    }
    MPI_Bcast(G0.adjncy.data(), M, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(G0.adjwgt.data(), M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank > 0)
        part.resize(G0.n);
    MPI_Bcast(part.data(), G0.n, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Build local subgraph ---
    std::vector<int> localGlo;
    for (int i = 0; i < G0.n; ++i)
        if (part[i] == rank)
            localGlo.push_back(i);
    int nloc = localGlo.size();

    std::unordered_map<int, int> glo2loc;
    glo2loc.reserve(nloc);
    for (int i = 0; i < nloc; ++i)
        glo2loc[localGlo[i]] = i;

    Graph localG(nloc);
    for (int gi : localGlo)
    {
        int ui = glo2loc[gi];
        for (int e = G0.xadj[gi]; e < G0.xadj[gi + 1]; ++e)
        {
            int gj = G0.adjncy[e];
            auto it = glo2loc.find(gj);
            if (it != glo2loc.end())
                localG.addEdge(ui, it->second, G0.adjwgt[e]);
        }
    }

    // --- Initial SSSP on local subgraph ---
    int SRC = 0;
    int localSrc = (part[SRC] == rank ? glo2loc[SRC] : -1);
    std::vector<real> dist;
    double t0 = MPI_Wtime();
    localDijkstra(localG, localSrc, dist);
    double t1 = MPI_Wtime();
    if (rank == 0)
        std::cout << "Initial Dijkstra (MPI rank 0): " << (t1 - t0) << " s\n";

    // --- Random dynamic update ---
    std::mt19937_64 rng(rank + 1);
    std::uniform_int_distribution<int> pick(0, nloc - 1);
    std::uniform_real_distribution<real> wr(1.0, 10.0);

    int u = pick(rng), v = pick(rng);
    while (v == u)
        v = pick(rng);
    real w = wr(rng);

    std::cout << "Rank " << rank
              << ": Insert edge (" << u << "," << v << ") w=" << w << "\n";
    localG.addEdge(u, v, w);
    std::vector<std::tuple<int, int, real>> inst, dels;
    inst.emplace_back(u, v, w);

    if (!localG.adj[u].empty())
    {
        auto pr = localG.adj[u][0];
        int dv = pr.first;
        real dw = pr.second;
        localG.removeEdge(u, dv);
        dels.emplace_back(u, dv, dw);
        std::cout << "Rank " << rank
                  << ": Delete edge (" << u << "," << dv << ") w=" << dw << "\n";
    }

    // --- Incremental update ---
    std::set<int> AD, A;
    std::vector<int> parent(nloc, -1);
    double t2 = MPI_Wtime();
    ProcessCE(localG, dels, inst, dist, parent, AD, A);
    UpdateAffectedVertices(localG, dist, parent, AD, A);
    double t3 = MPI_Wtime();

    std::cout << "Rank " << rank
              << ": Dynamic update time = " << (t3 - t2) << " s\n";

    // --- Optimized global reduction of first 100 distances ---
    const int K = 100;
    std::vector<real> myTop(K, INF), globalTop(K, INF);
    for (int i = 0; i < nloc; ++i)
    {
        int gi = localGlo[i];
        if (gi < K)
            myTop[gi] = dist[i];
    }
    MPI_Request req;
    MPI_Iallreduce(
        myTop.data(), globalTop.data(),
        K, MPI_DOUBLE, MPI_MIN,
        MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    if (rank == 0)
    {
        std::ofstream fout("dynamic_sssp_output.txt");
        fout << std::fixed;
        for (int i = 0; i < K; ++i)
        {
            if (globalTop[i] == INF)
                fout << i << " unreachable\n";
            else
                fout << i << " " << globalTop[i] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}