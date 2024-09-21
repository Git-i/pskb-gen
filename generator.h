#pragma once
#include "context.h"
#include <string>
class generator {
public:
    enum class input_mode {
        folder, panorama
    };
    generator(context& c) : ctx(c){}
    void set_input_mode(input_mode m) {mode = m;}
    void set_input(std::string i ) {input = i;}
    void set_input_ext(std::string i ) {input_ext = i;}
    void set_output(std::string o ) {output = o;}
    void generate();

private:
    input_mode mode;
    std::string input;
    std::string output;
    std::string input_ext;
    context& ctx;

    RHI::Ptr<RHI::Buffer> stagingBuffer;

    uint32_t width, height;
    RHI::Ptr<RHI::Texture> base;
    RHI::Ptr<RHI::Texture> ir;
    RHI::Ptr<RHI::Texture> pf;

private:
    void create_base();
    void create_ir(uint32_t ir_size);
    void create_pf();
};