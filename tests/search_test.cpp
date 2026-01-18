#include "index.hpp"
#include <iostream>
#include <chrono>
#include <memory>
#include <omp.h>
#include <boost/program_options.hpp>

using namespace std;
using namespace nns;

namespace opt = boost::program_options;


template <typename T>
void test(Paths & p, int k, float expand, int niter, int nthreads){

    auto efs = vector<uint32_t>{10, 11, 12, 13, 15, 18, 22, 26, 28, 35, 50, 60, 70, 80, 100, 128, 156, 192, 256};//, 298, 348, 400, 456, 512};

    auto index = new nns::Index<T>();

    index->loadindexDir(p.index_dir);

    index->set_num_threads(nthreads);
    index->set_search_k(k);
    index->set_expand(expand);
    index->initSearcher();

    auto queries = loader::load_queries<T>(p.query_path);
    
    auto gt = loader::load_groundtruth(p.gt_path);
    vector<vector<idx_t>> results(queries.N);
    vector<float> QPS(efs.size());
    vector<float> recalls(efs.size());
    vector<float> Hops(efs.size(), std::numeric_limits<float>::max());

    for(int it = 0; it < niter; it++) {
        #if defined DEBUG_VERBOSE
            cout << "------------------------------\n" 
                 << "ef,throughout,recall\n";
        #else
            printf("--------------------------------------\n               iter: %2d\n--------------------------------------\n", it);
            printf("efSearch, QueriesPerSecond, %10s\n", string("Recall@"+to_string(k)).c_str());
        #endif
        for(int ef_i = 0; ef_i < efs.size(); ef_i++){
            auto ef = efs[ef_i];
            index->n_comps = 0;
            auto s = chrono::high_resolution_clock::now();
            #pragma omp parallel for num_threads(index->num_threads())
            for(int i = 0; i < queries.N; i++){
                results[i] = index->search(queries[i], ef);
            }
            auto e = chrono::high_resolution_clock::now();
            auto duration = chrono::duration<double>(e - s).count();
            auto recall = utils::getRecall(results, gt, k);
            float nq_per_s = queries.N / duration;

            QPS[ef_i] = std::max(QPS[ef_i], nq_per_s);
            recalls[ef_i] = std::max(recalls[ef_i], recall);
            Hops[ef_i] = std::min(Hops[ef_i], (float)index->sum/queries.N);

            #if defined DEBUG_VERBOSE
                cout << ef << ',' << nq_per_s << ',' << recall << endl;
            #else
                printf("%8d, %16.1f, %10.5f \n", ef, nq_per_s, recall);
            #endif
        }
        if(p.out_path != "") {
            std::ofstream ofile(p.out_path);
            ofile << "efSearch,QueriesPerSecond,Recall@"+to_string(k) << ",nCalc" << endl;
            for(size_t ef_i = 0; ef_i < efs.size(); ++ef_i) {
                ofile << efs[ef_i] << ',' << QPS[ef_i] << ',' << recalls[ef_i] << ',' << Hops[ef_i] << endl;
            }
            ofile.close();
        }
    }
}


int main(int argc, char * argv[]){


    Paths p;
    int k, nthreads, niter;
    float expand;

    opt::options_description desc{"Arguments"};
    try
    {        
        desc.add_options()("help,h",         "Print information on arguments");

        opt::options_description required_configs("Required");

        required_configs.add_options()("index_dir",         opt::value<std::string>(&p.index_dir)->required(),                            "index directory");
        required_configs.add_options()("query_path",        opt::value<std::string>(&p.query_path)->required(),                         "Query file path: .*vecs or .*bin");
        required_configs.add_options()("gt_path",           opt::value<std::string>(&p.gt_path)->required(),                            "Ground truth: .ivecs or .ubin");
        required_configs.add_options()("topk,k",            opt::value<int>(&k)->required(),                                       "Top-k of nearest neighbor search");

        opt::options_description optional_configs("Optional");
        optional_configs.add_options()("out_path,o",        opt::value<std::string>(&p.out_path)->default_value(""),                    "QPS-Recall output path: .csv");
        optional_configs.add_options()("nthreads,t",        opt::value<int>(&nthreads)->default_value(1),                               "Num. of threads");        
        optional_configs.add_options()("iter,i",            opt::value<int>(&niter)->default_value(5),                                  "Num. of test iteration");
        optional_configs.add_options()("expand,e",          opt::value<float>(&expand)->default_value(1.0f),                            "efSearch expansion");


        desc.add(required_configs).add(optional_configs);
        opt::variables_map vm;
        opt::store(opt::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << endl;
            return 0;
        }
        opt::notify(vm);
    }
    catch(const std::exception & ex)
    {
        std::cerr << "Arguments error: " << ex.what() << std::endl;
        std::cerr << "Usage:\n";
        std::cout << desc << std::endl;
        return -1;
    }

    string datatype = nns::loader::getIndexDataType(p.index_dir);

    if(datatype == "float"){
        test<float>(p, k, expand, niter, nthreads);
    }else if(datatype == "uint8"){
        test<uint8_t>(p, k, expand, niter, nthreads);
    }else if(datatype == "int8"){
        test<int8_t>(p, k, expand, niter, nthreads);
    }

    return 0;
}

