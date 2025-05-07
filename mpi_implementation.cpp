#include <mpi.h>

#include <metis.h>

#include <chrono>

#include <iostream>

#include <fstream>

#include <vector>

#include <queue>

#include <limits>

#include <random>

#include <algorithm>

#include <unordered_map>

using real = double;

const real INF = std::numeric_limits<real>::infinity();

struct Edge
{
    int u, v;
    real w;
};

// Read CSR from file (expects exactly three columns: u v w)
/*void readCSR(const std::string &fname,
             std::vector<int> &xadj,
             std::vector<int> &adjncy,
             std::vector<real> &eweights,
             int &n)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::ifstream fin(fname);
    if (!fin) {
        if (rank == 0) std::cerr << "Cannot open " << fname << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int u, v;
    real w;
    std::vector<std::tuple<int,int,real>> edges;
    int maxv = -1;

    // Read three columns per line: u v w
    while (fin >> u >> v >> w) {
        // store both directions for an undirected CSR
        edges.emplace_back(u, v, w);
        edges.emplace_back(v, u, w);
        maxv = std::max({maxv, u, v});
    }
    fin.close();

    if (rank == 0 && !edges.empty()) {
        auto &first = edges.front();
        auto &last  = edges.back();
        std::cout << "First edge: "
                  << std::get<0>(first) << " - "
                  << std::get<1>(first) << " weight "
                  << std::get<2>(first) << "\n";
        std::cout << "Last edge: "
                  << std::get<0>(last) << " - "
                  << std::get<1>(last) << " weight "
                  << std::get<2>(last) << "\n";
    }

    n = maxv + 1;
    int M = edges.size();
    xadj.assign(n+1, 0);

    // count degrees
    for (auto &e : edges) {
        int uu = std::get<0>(e);
        xadj[uu+1]++;
    }
    // prefix‐sum
    for (int i = 1; i <= n; ++i) {
        xadj[i] += xadj[i-1];
    }

    adjncy.resize(M);
    eweights.resize(M);
    std::vector<int> ptr(xadj.begin(), xadj.end());

    // fill CSR arrays
    for (auto &e : edges) {
        int uu = std::get<0>(e);
        int vv = std::get<1>(e);
        real ww = std::get<2>(e);
        int pos = ptr[uu]++;
        adjncy[pos]  = vv;
        eweights[pos] = ww;
    }

    if (rank == 0) {
        std::cout << "Loaded graph: "
                  << n << " vertices, "
                  << M/2 << " edges\n";
    }
}
*/

void readCSR(const std::string &fname, std::vector<int> &xadj, std::vector<int> &adjncy, std::vector<real> &eweights, int &n)
{

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::ifstream fin(fname);

    if (!fin)
    {

        if (rank == 0)
            std::cerr << "Cannot open " << fname << "\n";

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int idx, u, v;

    real w;

    std::vector<Edge> edges;

    int maxv = -1;

    while (fin >> idx >> u >> v >> w)
    {

        edges.push_back({u, v, w});

        maxv = std::max({maxv, u, v});
    }

    fin.close();

    if (rank == 0 && !edges.empty())
    {

        std::cout << "First edge: " << edges.front().u << " - " << edges.front().v << " weight " << edges.front().w << "\n";

        std::cout << "Last edge: " << edges.back().u << " - " << edges.back().v << " weight " << edges.back().w << "\n";
    }

    int m = edges.size();

    if (rank == 0)
        std::cout << "Loaded graph: " << (maxv + 1) << " vertices, " << m << " edges\n";

    n = maxv + 1;

    xadj.assign(n + 1, 0);

    for (auto &e : edges)
    {

        xadj[e.u + 1]++;

        xadj[e.v + 1]++;
    }

    for (int i = 1; i <= n; ++i)
        xadj[i] += xadj[i - 1];

    adjncy.resize(2 * m);

    eweights.resize(2 * m);

    std::vector<int> pos = xadj;

    for (auto &e : edges)
    {

        int pu = pos[e.u]++;

        adjncy[pu] = e.v;
        eweights[pu] = e.w;

        int pv = pos[e.v]++;

        adjncy[pv] = e.u;
        eweights[pv] = e.w;
    }
}

// Standard Dijkstra on adjacency list

void dijkstra_local(const std::vector<std::vector<std::pair<int, real>>> &adj, int src, std::vector<real> &dist, std::vector<int> &parent)
{

    int N = adj.size();

    dist.assign(N, INF);

    parent.assign(N, -1);

    if (src >= 0)
    {

        dist[src] = 0;

        using P = std::pair<real, int>;

        std::priority_queue<P, std::vector<P>, std::greater<P>> pq;

        pq.emplace(0, src);

        while (!pq.empty())
        {

            auto [d, u] = pq.top();
            pq.pop();

            if (d > dist[u])
                continue;

            for (auto &pr : adj[u])
            {

                int v = pr.first;
                real w = pr.second;

                real nd = d + w;

                if (nd < dist[v])
                {

                    dist[v] = nd;

                    parent[v] = u;

                    pq.emplace(nd, v);
                }
            }
        }
    }
}

// Identify affected vertices

std::vector<int> identifyAffectedSubgraphs(const std::vector<int> &parent, const std::vector<real> &dist, const std::vector<Edge> &dels, const std::vector<Edge> &inst, const std::vector<std::vector<std::pair<int, real>>> &adj)
{

    std::vector<int> affected;

    std::unordered_map<int, bool> visited;

    // For deletions

    for (const auto &e : dels)
    {

        if (parent[e.v] == e.u && dist[e.v] != INF)
        {

            std::queue<int> q;

            q.push(e.v);

            while (!q.empty())
            {

                int v = q.front();
                q.pop();

                if (visited[v])
                    continue;

                visited[v] = true;

                affected.push_back(v);

                for (const auto &pr : adj[v])
                {

                    int u = pr.first;

                    if (parent[u] == v)
                        q.push(u);
                }
            }
        }
    }

    // For insertions

    for (const auto &e : inst)
    {

        if (dist[e.u] + e.w < dist[e.v])
        {

            affected.push_back(e.v);
        }
    }

    return affected;
}

// Extract subgraph CSR

void extractSubgraph(const std::vector<int> &affected, const std::vector<int> &xadj, const std::vector<int> &adjncy, const std::vector<real> &eweights, std::vector<int> &subXadj, std::vector<int> &subAdjncy, std::vector<real> &subEweights, std::unordered_map<int, int> &globalToLocal)
{

    int subN = affected.size();

    globalToLocal.clear();

    for (int i = 0; i < subN; ++i)
        globalToLocal[affected[i]] = i;

    subXadj.resize(subN + 1, 0);

    std::vector<std::vector<std::pair<int, real>>> subAdj(subN);

    for (int i = 0; i < subN; ++i)
    {

        int u = affected[i];

        for (int e = xadj[u]; e < xadj[u + 1]; ++e)
        {

            int v = adjncy[e];

            if (globalToLocal.count(v))
            {

                int lv = globalToLocal[v];

                subAdj[i].emplace_back(lv, eweights[e]);

                subXadj[i + 1]++;
            }
        }
    }

    for (int i = 1; i <= subN; ++i)
        subXadj[i] += subXadj[i - 1];

    subAdjncy.resize(subXadj[subN]);

    subEweights.resize(subXadj[subN]);

    std::vector<int> pos = subXadj;

    for (int i = 0; i < subN; ++i)
    {

        for (auto &pr : subAdj[i])
        {

            int j = pos[i]++;

            subAdjncy[j] = pr.first;

            subEweights[j] = pr.second;
        }
    }
}

// METIS partition for subgraph

std::vector<int> METIS_PartitionSubgraph(const std::vector<int> &subXadj, const std::vector<int> &subAdjncy, int subN, int np)
{

    idx_t nvtxs = subN, ncon = 1, objval;

    std::vector<idx_t> xadj_t(subXadj.begin(), subXadj.end());

    std::vector<idx_t> adjncy_t(subAdjncy.begin(), subAdjncy.end());

    std::vector<idx_t> part_t(subN);

    METIS_PartGraphKway(&nvtxs, &ncon, xadj_t.data(), adjncy_t.data(), NULL, NULL, NULL, &np, NULL, NULL, NULL, &objval, part_t.data());

    std::vector<int> part(subN);

    for (int i = 0; i < subN; ++i)
        part[i] = part_t[i];

    return part;
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int rank, np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (rank == 0)
        std::cout << "Using " << np << " MPI processes\n";

    // 1) Read CSR on rank 0 and broadcast

    std::vector<int> xadj, adjncy;

    std::vector<real> eweights;

    int n;

    if (rank == 0)
        readCSR("graphnew.txt", xadj, adjncy, eweights, n);

    // broadcast number of vertices

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- UPDATED: broadcast full-graph adjacency lengths ---

    int M = 0;

    if (rank == 0)
    {

        M = static_cast<int>(adjncy.size());
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {

        xadj.resize(n + 1);

        adjncy.resize(M);

        eweights.resize(M);
    }

    // broadcast the actual CSR arrays

    MPI_Bcast(xadj.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(adjncy.data(), M, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(eweights.data(), M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- end UPDATED section ---

    // build full adjacency list

    std::vector<std::vector<std::pair<int, real>>> adj(n);

    for (int u = 0; u < n; ++u)
    {

        for (int e = xadj[u]; e < xadj[u + 1]; ++e)
        {

            adj[u].emplace_back(adjncy[e], eweights[e]);
        }
    }

    // 2) Initial SSSP on rank 0

    int global_src = 0;

    std::vector<real> dist(n, INF);

    std::vector<int> parent(n, -1);

    if (rank == 0)
    {

        auto t0 = std::chrono::high_resolution_clock::now();

        dijkstra_local(adj, global_src, dist, parent);

        auto t1 = std::chrono::high_resolution_clock::now();

        std::cout << "Initial Dijkstra time: "

                  << std::chrono::duration<double>(t1 - t0).count()

                  << " s\n";
    }

    MPI_Bcast(dist.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(parent.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    // 3) Simulate dynamic change on rank 0 and broadcast

    std::vector<Edge> dels, inst;

    if (rank == 0)
    {

        std::mt19937_64 rng(std::random_device{}());

        std::uniform_int_distribution<int> pick(0, n - 1);

        int u = pick(rng), v = pick(rng);

        while (v == u)
            v = pick(rng);

        real w = std::uniform_real_distribution<real>(1, 10)(rng);

        inst.push_back({u, v, w});

        std::cout << "Inserted edge: (" << u << "," << v << ") weight " << w << "\n";

        int del_v = adjncy[xadj[u]];

        dels.push_back({u, del_v, eweights[xadj[u]]});

        std::cout << "Deleted edge: (" << u << "," << del_v << ") weight "

                  << eweights[xadj[u]] << "\n";
    }

    int kd = dels.size(), ki = inst.size();

    MPI_Bcast(&kd, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&ki, 1, MPI_INT, 0, MPI_COMM_WORLD);

    dels.resize(kd);

    inst.resize(ki);

    MPI_Bcast(dels.data(), kd * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Bcast(inst.data(), ki * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 4) Identify affected vertices and broadcast

    std::vector<int> affected =

        identifyAffectedSubgraphs(parent, dist, dels, inst, adj);

    int subN = affected.size();

    MPI_Bcast(&subN, 1, MPI_INT, 0, MPI_COMM_WORLD);

    affected.resize(subN);

    MPI_Bcast(affected.data(), subN, MPI_INT, 0, MPI_COMM_WORLD);

    // 5) Extract subgraph on rank 0

    std::vector<int> subXadj, subAdjncy;

    std::vector<real> subEweights;

    std::unordered_map<int, int> globalToLocal;

    if (rank == 0)
    {

        extractSubgraph(affected, xadj, adjncy, eweights,

                        subXadj, subAdjncy, subEweights, globalToLocal);
    }

    // --- UPDATED: broadcast subgraph CSR lengths and data ---

    int subXn = 0, subEn = 0;

    if (rank == 0)
    {

        subXn = static_cast<int>(subXadj.size());

        subEn = static_cast<int>(subAdjncy.size());
    }

    MPI_Bcast(&subXn, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&subEn, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {

        subXadj.resize(subXn);

        subAdjncy.resize(subEn);

        subEweights.resize(subEn);
    }

    MPI_Bcast(subXadj.data(), subXn, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(subAdjncy.data(), subEn, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(subEweights.data(), subEn, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- end UPDATED section ---

    // 6) Partition subgraph with METIS and broadcast

    std::vector<int> subPart;

    if (rank == 0)
    {

        subPart = METIS_PartitionSubgraph(subXadj, subAdjncy, subN, np);
    }

    subPart.resize(subN);

    MPI_Bcast(subPart.data(), subN, MPI_INT, 0, MPI_COMM_WORLD);

    // 7) Build local subgraph adjacency list

    std::vector<int> local_v;

    for (int i = 0; i < subN; ++i)

        if (subPart[i] == rank)
            local_v.push_back(i);

    int ln = local_v.size();

    std::unordered_map<int, int> subG2L;

    for (int i = 0; i < ln; ++i)

        subG2L[local_v[i]] = i;

    std::vector<std::vector<std::pair<int, real>>> subAdjLocal(ln);

    for (int i = 0; i < subN; ++i)
    {

        if (subPart[i] != rank)
            continue;

        int lu = subG2L[i];

        for (int e = subXadj[i]; e < subXadj[i + 1]; ++e)
        {

            subAdjLocal[lu].emplace_back(subAdjncy[e], subEweights[e]);
        }
    }

    // 8) Apply SSSP to local subgraph

    std::vector<real> subDistLocal(ln, INF);

    std::vector<int> subParentLocal(ln, -1);

    // NEW (use the map you actually built):

    int lsrc = -1;

    if (globalToLocal.count(global_src) && subG2L.count(globalToLocal[global_src]))
    {

        int subIdx = globalToLocal[global_src]; // map global→subgraph index

        lsrc = subG2L[subIdx]; // map subgraph→local index
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    dijkstra_local(subAdjLocal, lsrc, subDistLocal, subParentLocal);

    auto t5 = std::chrono::high_resolution_clock::now();

    if (rank == 0)

        std::cout << "Dynamic update time: "

                  << std::chrono::duration<double>(t5 - t4).count()

                  << " s\n";

    // 9) Send updated distances back to rank 0

    std::vector<int> updated_global_indices;

    std::vector<real> updated_distances;

    for (int i = 0; i < ln; ++i)
    {

        int global_u = affected[local_v[i]];

        updated_global_indices.push_back(global_u);

        updated_distances.push_back(subDistLocal[i]);
    }

    int num_updated = updated_global_indices.size();

    MPI_Send(&num_updated, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    if (num_updated > 0)
    {

        MPI_Send(updated_global_indices.data(), num_updated, MPI_INT, 0, 1, MPI_COMM_WORLD);

        MPI_Send(updated_distances.data(), num_updated, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    // 10) On rank 0, receive and merge updates

    if (rank == 0)
    {

        for (int p = 0; p < np; ++p)
        {

            int nu;

            MPI_Recv(&nu, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (nu > 0)
            {

                std::vector<int> idx(nu);

                std::vector<real> d2(nu);

                MPI_Recv(idx.data(), nu, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Recv(d2.data(), nu, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < nu; ++i)

                    dist[idx[i]] = d2[i];
            }
        }

        // 11) Write first 100 nodes to file

        std::ofstream fout("dynamic_sssp_openmpi_output.txt");

        fout << std::fixed;

        for (int i = 0; i < 100 && i < n; ++i)
        {

            if (dist[i] == INF)
                fout << i << " unreachable\n";

            else
                fout << i << " " << dist[i] << "\n";
        }

        fout.close();
    }

    MPI_Finalize();

    return 0;
}