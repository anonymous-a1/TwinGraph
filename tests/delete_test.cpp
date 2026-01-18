#include "index.hpp"
#include <boost/program_options.hpp>

namespace opt = boost::program_options;

template <typename T>
void test(nns::Paths p, size_t del_start, size_t del_end, bool if_rejuv){
    auto index = new nns::Index<T>();
    index->loadindexDir(p.index_dir);
    
    auto s = std::chrono::high_resolution_clock::now();
    for(int i = del_start; i < del_end; i++){ 
        index->deleteVertex(i);
    }
    if(if_rejuv) index->rejuvenation();
    auto e = std::chrono::high_resolution_clock::now();
    std::cout << "delete timecost: " << std::chrono::duration<double>(e - s).count() << std::endl;

    index->printAvgEdge();
    if(p.out_dir != "") index->saveIndexDir(p.out_dir);
}


int main(int argc, char * argv[]){

    nns::Paths p;
    size_t del_start, del_end;
    bool rejuv_now = false;
    int nthreads;
    
    opt::options_description desc{"Arguments"};
    try
    {        
        desc.add_options()("help,h",         "Print information on arguments");

        opt::options_description required_configs("Required");

        required_configs.add_options()("index_dir",         opt::value<std::string>(&p.index_dir)->required(),                          "Input index directory");
        required_configs.add_options()("del_start",         opt::value<size_t>(&del_start)->required(),                                 "Start id of delete set (included)");
        required_configs.add_options()("del_end",           opt::value<size_t>(&del_end)->required(),                                   "End id of delete set (not included)");
        
        opt::options_description optional_configs("Optional");
        optional_configs.add_options()("out_dir",           opt::value<std::string>(&p.out_dir)->default_value(""),                     "Output index directory");
        optional_configs.add_options()("rejuv_now",         opt::bool_switch(&rejuv_now)->default_value(false),                         "Use this to immediately rejuvenate after deletion");     
        optional_configs.add_options()("nthreads,t",        opt::value<int>(&nthreads)->default_value(omp_get_max_threads()),           "Num. of threads");     
        
        desc.add(required_configs).add(optional_configs);
        opt::variables_map vm;
        opt::store(opt::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
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

    omp_set_num_threads(nthreads);
    std::string datatype = nns::loader::getIndexDataType(p.index_dir);

    if(datatype == "float"){
        test<float>(p, del_start, del_end, rejuv_now);
    }else if(datatype == "uint8"){
        test<uint8_t>(p, del_start, del_end, rejuv_now);
    }else if(datatype == "int8"){
        test<int8_t>(p, del_start, del_end, rejuv_now);
    }

}