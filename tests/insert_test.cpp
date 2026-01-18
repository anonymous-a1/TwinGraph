#include "index.hpp"
#include <boost/program_options.hpp>

namespace opt = boost::program_options;

template <typename T>
void test(nns::Paths p, size_t ins_start, size_t ins_end, size_t label_offset = 0){
    auto index = new nns::Index<T>();
    index->loadindexDir(p.index_dir);
    index->initSearcher();
    auto data = nns::loader::load_queries<T>(p.data_path);

    std::cout << "current index status: current num = " << index->getCurrNum() << ", maximum num = " << index->getMaxNum() << std::endl;
    std::cout << "get " << data.N << " vectors to add\n";
    std::cout << "start update index...\n";

    if(label_offset = 0) label_offset = index->getCurrNum();
    {
    auto s = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for(int i = ins_start; i < ins_end; i++){
        index->addVertex(data[i], i + label_offset);
    }
    auto e = std::chrono::steady_clock::now();
    std::cout << "insert timecost: " << std::chrono::duration<double>(e - s).count() << std::endl;
    }
    index->printAvgEdge(); 

    if(p.out_dir != "") index->saveIndexDir(p.out_dir);
}

int main(int argc, char * argv[]){

    nns::Paths p;
    size_t ins_start, ins_end, label_offset = 0;
    int nthreads;
    
    opt::options_description desc{"Arguments"};
    try
    {        
        desc.add_options()("help,h",         "Print information on arguments");

        opt::options_description required_configs("Required");
        required_configs.add_options()("index_dir",         opt::value<std::string>(&p.index_dir)->required(),                          "Input index directory");
        required_configs.add_options()("data_path",         opt::value<std::string>(&p.data_path)->required(),                          "File path of vectors to be inserted: .*vecs or .*bin");
        required_configs.add_options()("ins_start",         opt::value<size_t>(&ins_start)->required(),                                 "Start id of input basedata (included)");
        required_configs.add_options()("ins_end",           opt::value<size_t>(&ins_end)->required(),                                   "End id of input basedata (not included)");

        opt::options_description optional_configs("Optional");
        optional_configs.add_options()("label_offset",      opt::value<size_t>(&label_offset),                                          "Offset of insert vector label, default: current vector num");
        optional_configs.add_options()("out_dir",           opt::value<std::string>(&p.out_dir)->default_value(""),                     "Output index directory");
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
        test<float>(p, ins_start, ins_end, label_offset);
    }else if(datatype == "uint8"){
        test<uint8_t>(p, ins_start, ins_end, label_offset);
    }else if(datatype == "int8"){
        test<int8_t>(p, ins_start, ins_end, label_offset);
    }

    return 0;
}
