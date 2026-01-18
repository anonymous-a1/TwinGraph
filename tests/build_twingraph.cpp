#include "index.hpp"
#include <boost/program_options.hpp>

namespace opt = boost::program_options;

template <typename T>
void test(nns::Paths & p, nns::build_params & bp, size_t maxnum){

    auto index = new nns::Index<T>();
    index->initData(p.data_path, maxnum);

    index->buildTwinGraph(bp);

    std::cout << "Mem usage: " << MemInfo:: getMemoryUsage() << " MB\n";

    if(p.out_dir != "") index->saveIndexDir(p.out_dir);
    if(p.out_knng != "") index->saveKNNG(p.out_knng);
}


int main(int argc, char * argv[]){

    nns::Paths p;
    nns::build_params bp;
    std::string datatype;
    int nthreads, niter, qdim{0};
    size_t maxnum{0};

    opt::options_description desc{"Arguments"};
    try
    {        
        desc.add_options()("help,h",         "Print information on arguments");

        opt::options_description required_configs("Required");
        required_configs.add_options()("data_path",         opt::value<std::string>(&p.data_path)->required(),                          "Base vector file path: .*vecs|.*bin");
        required_configs.add_options()("data_type",         opt::value<std::string>(&datatype)->required(),                             "Input base data type: float|int8|uint8");
        required_configs.add_options()("K,K",               opt::value<int>(&bp.K)->required(),                                         "Max neighbor num of kNN graph");
        required_configs.add_options()("alpha,a",           opt::value<float>(&bp.alpha)->required(),                                   "Parameter alpha of graph diversify");
        required_configs.add_options()("R,R",               opt::value<int>(&bp.R)->required(),                                         "Max out degree of index");

        opt::options_description optional_configs("Optional");
        optional_configs.add_options()("maxiter,i",         opt::value<int>(&bp.MaxIter)->default_value(100),                           "Max iteration");
        optional_configs.add_options()("out_dir",           opt::value<std::string>(&p.out_dir),                                        "Output index path");        
        optional_configs.add_options()("out_knng",          opt::value<std::string>(&p.out_knng),                                       "Output knn graph path: .ivecs|ubin");
        optional_configs.add_options()("nthreads,t",        opt::value<int>(&nthreads)->default_value(omp_get_max_threads()),           "Num. of threads ");
        optional_configs.add_options()("maxnum",            opt::value<size_t>(&maxnum),                                                "Max vertex num, default: current input data size");

        desc.add(required_configs).add(optional_configs);
        opt::variables_map vm;
        opt::store(opt::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return 0;
        }
        opt::notify(vm);
    }catch(const std::exception & ex){
        std::cerr << "Arguments error: " << ex.what() << std::endl;
        std::cerr << "Usage:\n";
        std::cout << desc << std::endl;
        return -1;
    }

    omp_set_num_threads(nthreads);

    if(datatype == "float"){
        test<float>(p, bp, maxnum);
    }else if(datatype == "uint8"){
        test<uint8_t>(p, bp, maxnum);
    }else if(datatype == "int8"){
        test<int8_t>(p, bp, maxnum);
    }

    return 0;
}
