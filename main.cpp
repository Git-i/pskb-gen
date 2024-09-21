#include "generator.h"
#include <csignal>
#include <argparse/argparse.hpp>
generator::input_mode input_mode_from_string(std::string_view mode) noexcept(false)
{
    if(mode == "panorama") return generator::input_mode::panorama;
    if(mode == "folder") return generator::input_mode::folder;
    throw std::runtime_error("Invalid mode");
}
int main(int argc, char** argv)
{
    ::setenv("DXVK_WSI_DRIVER", "GLFW", 1);
    try{
    argparse::ArgumentParser parser("pskb-gen");
    parser.add_argument("file");
    parser.add_argument("-im", "--input-mode")
        .help("Input mode, -im (panorama | folder)")
        .required();
    parser.add_argument("-o", "--output")
        .help("Output file, -o <filename>")
        .required();
    parser.add_argument("-ext")
        .help("file extension when using folder input mode");
    parser.parse_args(argc, argv);
    context ctx;
    ctx.initialize();
    generator g(ctx);
    const auto im = input_mode_from_string(parser.get("-im"));
    g.set_input_mode(im);
    g.set_input(parser.get("file"));
    if (im != generator::input_mode::panorama)
    {
        g.set_input_ext(parser.get("-ext"));
    }
    g.set_output(parser.get("-o"));
    g.generate();
    } catch(const std::exception& e)
    {
        std::cout << e.what() << std::endl;
        std::raise(SIGTRAP);
    }
}