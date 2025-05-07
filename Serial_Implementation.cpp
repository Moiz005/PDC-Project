#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <tuple>
#include <sstream>
#include <random>
#include <algorithm>
#include <set>

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
    // remove exactly one copy of (u,v) and (v,u) if present
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

// Read 4-column file "idx u v w", ignore idx
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
    edges.reserve(1 << 20);
    int edge_count = 0;
    while (fin >> idx >> u >> v >> w)
    {
        edges.push_back({u, v, w});
        maxv = std::max({maxv, u, v});
        edge_count++;
        if (edge_count % 1000 == 0)
        {
            // std::cout << "Read " << edge_count << " edges" << std::endl;
        }
    }
    fin.close();
    if (!edges.empty())
    {
        std::cout << "First edge: " << edges[0].u << " - " << edges[0].v << " weight " << edges[0].w << std::endl;
        std::cout << "Last edge: " << edges.back().u << " - " << edges.back().v << " weight " << edges.back().w << std::endl;
    }
    Graph G(maxv + 1);
    for (auto &e : edges)
        G.addEdge(e.u, e.v, e.w);
    std::cout << "Loaded graph: " << G.n << " vertices, " << edges.size() << " edges\n";
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

// Algorithm 2
void ProcessCE(const Graph &G,
               const std::vector<Edge> &Del,
               const std::vector<Edge> &Inst,
               std::vector<real> &dist,
               std::vector<int> &parent,
               std::set<int> &AD,
               std::set<int> &A)
{
    AD.clear();
    A.clear();
    for (auto &e : Del)
    {
        int u = e.u, v = e.v;
        if (parent[u] == v || parent[v] == u)
        {
            int y = (dist[u] > dist[v] ? u : v);
            dist[y] = INF;
            AD.insert(y);
            A.insert(y);
        }
    }
    for (auto &e : Inst)
    {
        int u = e.u, v = e.v;
        real w = e.w;
        int x = (dist[u] > dist[v] ? v : u), y = (x == u ? v : u);
        if (dist[x] < INF && dist[x] + w < dist[y])
        {
            dist[y] = dist[x] + w;
            parent[y] = x;
            A.insert(y);
        }
    }
}

// Algorithm 3, optimized with sets
void UpdateAffectedVertices(const Graph &G,
                            std::vector<real> &dist, std::vector<int> &parent,
                            std::set<int> &AD, std::set<int> &A)
{
    int N = G.n;
    // deletions
    while (!AD.empty())
    {
        std::set<int> nextD;
        for (int v : AD)
        {
            for (int c = 0; c < N; ++c)
                if (parent[c] == v)
                {
                    dist[c] = INF;
                    nextD.insert(c);
                    A.insert(c);
                }
        }
        AD.swap(nextD);
    }
    // insert/update
    while (!A.empty())
    {
        std::set<int> nextA;
        for (int v : A)
        {
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
        }
        A.swap(nextA);
    }
}

int main()
{
    std::mt19937_64 rng(std::random_device{}());
    const std::string fname = "california.txt";
    const int SRC = 0;

    // 1) initial
    auto t0 = std::chrono::high_resolution_clock::now();
    Graph G = readGraph(fname);
    std::vector<real> dist;
    std::vector<int> parent;
    std::cout << "Applying SSSP ALGORITHM" << std::endl;
    dijkstra(G, SRC, dist, parent);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Initial Dijkstra time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    std::cout << "Applying it dynamically\n";

    // 2) simulate dynamic changes: insert and delete edges
    std::uniform_int_distribution<int> pick(0, G.n - 1);
    std::vector<Edge> dels, inst;

    // Example: insert a new edge
    int u = pick(rng), v = pick(rng);
    while (v == u)
        v = pick(rng);
    real w = std::uniform_real_distribution<real>(1.0, 10.0)(rng);
    G.addEdge(u, v, w);
    inst.push_back({u, v, w});
    std::cout << "Inserted edge: (" << u << "," << v << ") with weight " << w << "\n";

    // Example: delete an existing edge
    if (!G.adj[u].empty())
    {
        auto &e = G.adj[u][0];
        int del_v = e.first;
        real del_w = e.second;
        G.removeEdge(u, del_v);
        dels.push_back({u, del_v, del_w});
        std::cout << "Deleted edge: (" << u << "," << del_v << ") with weight " << del_w << "\n";
    }

    // 3) dynamic SSSP update
    std::set<int> AD, A;
    auto t4 = std::chrono::high_resolution_clock::now();
    ProcessCE(G, dels, inst, dist, parent, AD, A);
    UpdateAffectedVertices(G, dist, parent, AD, A);
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "Dynamic update time: "
              << std::chrono::duration<double>(t5 - t4).count()
              << " s\n";

    // 4) verify with full Dijkstra
    std::vector<real> dist2;
    std::vector<int> parent2;
    auto t6 = std::chrono::high_resolution_clock::now();
    dijkstra(G, SRC, dist2, parent2);
    auto t7 = std::chrono::high_resolution_clock::now();
    std::cout << "Full Dijkstra time: "
              << std::chrono::duration<double>(t7 - t6).count()
              << " s\n";

    // 5) check if dist and dist2 match
    bool match = true;
    for (int i = 0; i < G.n; ++i)
    {
        if (dist[i] != dist2[i])
        {
            match = false;
            std::cout << "Mismatch at vertex " << i << ": dynamic=" << dist[i] << ", full=" << dist2[i] << "\n";
            break;
        }
    }
    if (match)
        std::cout << "Dynamic update correct\n";
    else
        std::cout << "Dynamic update incorrect\n";

    // 6) output first 100 values
    std::ofstream fout("dynamic_sssp_output.txt");
    fout << std::fixed;
    for (int i = 0; i < std::min(G.n, 100); ++i)
    {
        if (dist[i] == INF)
            fout << i << " unreachable\n";
        else
            fout << i << " " << dist[i] << "\n";
    }
    return 0;
}