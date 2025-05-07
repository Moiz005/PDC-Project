#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <random>
#include <algorithm>
#include <omp.h> // OpenMP

using real = double;
const real INF = std::numeric_limits<real>::infinity();

struct Edge
{
    int u, v;
    real w;
};

class Graph
{
public:
    int n;
    std::vector<std::vector<std::pair<int, real>>> adj;
    Graph(int N = 0) : n(N), adj(N) {}

    void addEdge(int u, int v, real w)
    {
        if (u < 0 || u >= n || v < 0 || v >= n)
            return;
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    void removeEdge(int u, int v)
    {
        auto &A = adj[u];
        for (auto it = A.begin(); it != A.end(); ++it)
        {
            if (it->first == v)
            {
                A.erase(it);
                break;
            }
        }
        auto &B = adj[v];
        for (auto it = B.begin(); it != B.end(); ++it)
        {
            if (it->first == u)
            {
                B.erase(it);
                break;
            }
        }
    }
};

// Read file "idx u v w" into Graph and print first/last edge info
Graph readGraph(const std::string &fname)
{
    std::ifstream fin(fname);
    if (!fin)
    {
        std::cerr << "Cannot open " << fname << "\n";
        std::exit(1);
    }
    int idx, u, v;
    real w;
    int maxv = -1;
    std::vector<Edge> edges;
    while (fin >> idx >> u >> v >> w)
    {
        edges.push_back({u, v, w});
        maxv = std::max({maxv, u, v});
    }
    fin.close();
    if (!edges.empty())
    {
        std::cout << "First edge: "
                  << edges.front().u << " - " << edges.front().v
                  << " weight " << edges.front().w << "\n";
        std::cout << "Last edge: "
                  << edges.back().u << " - " << edges.back().v
                  << " weight " << edges.back().w << "\n";
    }
    Graph G(maxv + 1);
    for (auto &e : edges)
        G.addEdge(e.u, e.v, e.w);
    std::cout << "Loaded graph: " << G.n << " vertices, "
              << edges.size() << " edges\n";
    return G;
}

// Standard Dijkstra
void dijkstra(const Graph &G, int src,
              std::vector<real> &dist, std::vector<int> &parent)
{
    int N = G.n;
    dist.assign(N, INF);
    parent.assign(N, -1);
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
        for (auto &pr : G.adj[u])
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

// Process changes (Algorithm 2) with OpenMP
void ProcessCE(const Graph &G,
               const std::vector<Edge> &Del,
               const std::vector<Edge> &Inst,
               std::vector<real> &dist,
               std::vector<int> &parent,
               std::vector<char> &in_AD,
               std::vector<char> &in_A,
               std::vector<int> &AD,
               std::vector<int> &A)
{
    int N = G.n;
    std::fill(in_AD.begin(), in_AD.end(), 0);
    std::fill(in_A.begin(), in_A.end(), 0);
    AD.clear();
    A.clear();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)Del.size(); ++i)
    {
        int u = Del[i].u, v = Del[i].v;
        if (parent[u] == v || parent[v] == u)
        {
            int y = (dist[u] > dist[v] ? u : v);
            dist[y] = INF;
#pragma omp critical
            {
                if (!in_AD[y])
                {
                    in_AD[y] = 1;
                    AD.push_back(y);
                }
                if (!in_A[y])
                {
                    in_A[y] = 1;
                    A.push_back(y);
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)Inst.size(); ++i)
    {
        int u = Inst[i].u, v = Inst[i].v;
        real w = Inst[i].w;
        int x = (dist[u] > dist[v] ? v : u), y = (x == u ? v : u);
        real dx = dist[x];
        if (dx < INF && dx + w < dist[y])
        {
#pragma omp critical
            {
                dist[y] = dx + w;
                parent[y] = x;
                if (!in_A[y])
                {
                    in_A[y] = 1;
                    A.push_back(y);
                }
            }
        }
    }
}

// Update affected vertices (Algorithm 3) with OpenMP
void UpdateAffectedVertices(const Graph &G,
                            std::vector<real> &dist, std::vector<int> &parent,
                            std::vector<char> &in_AD, std::vector<char> &in_A,
                            std::vector<int> &AD, std::vector<int> &A)
{
    int N = G.n;
    std::vector<char> in_nextD(N), in_nextA(N);
    std::vector<int> nextD, nextA;

    while (!AD.empty())
    {
        std::fill(in_nextD.begin(), in_nextD.end(), 0);
        nextD.clear();
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)AD.size(); ++i)
        {
            int v = AD[i];
            for (int c = 0; c < N; ++c)
            {
                if (parent[c] == v)
                {
                    dist[c] = INF;
#pragma omp critical
                    {
                        if (!in_nextD[c])
                        {
                            in_nextD[c] = 1;
                            nextD.push_back(c);
                        }
                        if (!in_A[c])
                        {
                            in_A[c] = 1;
                            A.push_back(c);
                        }
                    }
                }
            }
        }
        AD.swap(nextD);
    }

    while (!A.empty())
    {
        std::fill(in_nextA.begin(), in_nextA.end(), 0);
        nextA.clear();
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)A.size(); ++i)
        {
            int v = A[i];
            for (auto &pr : G.adj[v])
            {
                int n = pr.first;
                real w = pr.second;
                real dv = dist[v], dn = dist[n];
                if (dv < INF && dv + w < dn)
                {
#pragma omp critical
                    {
                        dist[n] = dv + w;
                        parent[n] = v;
                        if (!in_nextA[n])
                        {
                            in_nextA[n] = 1;
                            nextA.push_back(n);
                        }
                    }
                }
                if (dn < INF && dn + w < dv)
                {
#pragma omp critical
                    {
                        dist[v] = dn + w;
                        parent[v] = n;
                        if (!in_nextA[v])
                        {
                            in_nextA[v] = 1;
                            nextA.push_back(v);
                        }
                    }
                }
            }
        }
        A.swap(nextA);
    }
}

int main(int argc, char **argv)
{
    int num_threads = 8;
    if (argc > 1)
        num_threads = std::stoi(argv[1]);
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " OpenMP threads\n";

    const std::string fname = "california.txt";
    const int SRC = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    Graph G = readGraph(fname);
    std::vector<real> dist, dist2;
    std::vector<int> parent, parent2;
    std::cout << "Applying SSSP ALGORITHM\n";
    dijkstra(G, SRC, dist, parent);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Initial Dijkstra time: "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    std::cout << "Applying it dynamically\n";
    // simulate changes
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> pick(0, G.n - 1);
    std::vector<Edge> dels, inst;
    int u = pick(rng), v = pick(rng);
    while (v == u)
        v = pick(rng);
    real w = std::uniform_real_distribution<real>(1.0, 10.0)(rng);
    G.addEdge(u, v, w);
    inst.push_back({u, v, w});
    std::cout << "Inserted edge: (" << u << "," << v << ") with weight " << w << "\n";

    if (!G.adj[u].empty())
    {
        auto &e = G.adj[u][0];
        G.removeEdge(u, e.first);
        dels.push_back({u, e.first, e.second});
        std::cout << "Deleted edge: (" << u << "," << e.first << ") with weight " << e.second << "\n";
    }

    std::vector<char> in_AD(G.n), in_A(G.n);
    std::vector<int> AD, A;
    auto t4 = std::chrono::high_resolution_clock::now();
    ProcessCE(G, dels, inst, dist, parent, in_AD, in_A, AD, A);
    UpdateAffectedVertices(G, dist, parent, in_AD, in_A, AD, A);
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "Dynamic update time: "
              << std::chrono::duration<double>(t5 - t4).count() << " s\n";

    auto t6 = std::chrono::high_resolution_clock::now();
    dijkstra(G, SRC, dist2, parent2);
    auto t7 = std::chrono::high_resolution_clock::now();
    std::cout << "Full Dijkstra time: "
              << std::chrono::duration<double>(t7 - t6).count() << " s\n";

    bool match = true;
    for (int i = 0; i < G.n; ++i)
    {
        if (dist[i] != dist2[i])
        {
            match = false;
            break;
        }
    }
    std::cout << (match ? "Dynamic update correct\n" : "Dynamic update incorrect\n");

    // write first 100 distances to file
    std::ofstream fout("dynamic_omp_output.txt");
    fout << std::fixed;
    for (int i = 0; i < std::min(G.n, 100); ++i)
    {
        if (dist[i] == INF)
            fout << i << " unreachable\n";
        else
            fout << i << " " << dist[i] << "\n";
    }
    fout.close();
    return 0;
}